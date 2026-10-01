"""Step 2 - Log-log regression of biogas on herd size, k-fold CV vs a Monte Carlo baseline."""
import pandas as pd, numpy as np, json
import statsmodels.formula.api as smf
from sklearn.model_selection import RepeatedKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
op = pd.read_pickle("data/processed/op.pkl"); d = op.copy()
for c in ["Cattle", "Dairy", "Swine"]: d[c] = d[c].fillna(0)
d["N"] = d.Dairy + d.Cattle + d.Swine
d = d[(d.Biogas > 0) & (d.N > 0)].copy()
d["lnB"] = np.log(d.Biogas); d["lnN"] = np.log(d.N)
def dgroup(s):
    if s == "Covered Lagoon": return "CL"
    if s == "Complete Mix": return "CM"
    if "Plug Flow" in str(s): return "PF"
    return "Other"
d["D"] = d["Digester Type"].map(dgroup)
d["A"] = np.where(d.Swine > d.Dairy + d.Cattle, "Swine", "Cattle")
d["Codig"] = d["Co-Digestion"].notna().astype(int)
F2 = "lnB ~ lnN + C(D, Treatment('CL')) + C(A, Treatment('Cattle')) + Codig"
R = {"n_multi": len(d), "counts_D": d.D.value_counts().to_dict(), "counts_A": d.A.value_counts().to_dict(), "n_codig": int(d.Codig.sum())}
m1 = smf.ols("lnB ~ lnN", d).fit(); m2 = smf.ols(F2, d).fit()
R["m1"] = dict(params=m1.params.to_dict(), bse=m1.bse.to_dict(), R2=m1.rsquared, adjR2=m1.rsquared_adj, aic=m1.aic)
R["m2"] = dict(params=m2.params.to_dict(), bse=m2.bse.to_dict(), pvalues=m2.pvalues.to_dict(), R2=m2.rsquared,
               adjR2=m2.rsquared_adj, aic=m2.aic, ci=m2.conf_int().T.to_dict())
open("results/tables/m2_summary.txt", "w").write(m2.summary().as_text())
rkf = RepeatedKFold(n_splits=10, n_repeats=20, random_state=1); rng = np.random.default_rng(7)
res = {k: {"rmse": [], "mae": [], "mape": [], "r2": []} for k in ["MC baseline", "M1 simple", "M2 multiple"]}
for tr, te in rkf.split(d):
    a, b = d.iloc[tr], d.iloc[te]; preds = {}
    preds["MC baseline"] = rng.choice(a.lnB.values, size=len(b))
    preds["M1 simple"] = smf.ols("lnB ~ lnN", a).fit().predict(b)
    try:
        preds["M2 multiple"] = smf.ols(F2, a).fit().predict(b)
    except Exception:
        continue
    for k, p in preds.items():
        p = np.asarray(p); e = b.lnB.values - p
        res[k]["rmse"].append(np.sqrt(np.mean(e ** 2))); res[k]["mae"].append(np.mean(np.abs(e)))
        res[k]["mape"].append(np.mean(np.abs(np.exp(p) - b.Biogas.values) / b.Biogas.values) * 100)
        res[k]["r2"].append(1 - np.sum(e ** 2) / np.sum((b.lnB - b.lnB.mean()) ** 2))
cv = pd.DataFrame({k: {m: np.nanmean(v) for m, v in r.items()} for k, r in res.items()}).T
cv_sd = pd.DataFrame({k: {m: np.nanstd(v) for m, v in r.items()} for k, r in res.items()}).T
cv.to_csv("results/tables/cv_metrics.csv"); R["cv"] = cv.to_dict(); R["cv_sd"] = cv_sd.to_dict()
dd = d[d.A == "Cattle"]; res1 = smf.ols("lnB ~ lnN", dd).fit().resid
np.save("results/tables/resid_m1.npy", res1.values); R["resid_sd_dairy"] = float(res1.std())
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].scatter(m2.fittedvalues, m2.resid, s=10, alpha=.6, color="#1f6f8b"); ax[0].axhline(0, color="k", lw=.8)
ax[0].set_xlabel("Fitted ln B (M2)"); ax[0].set_ylabel("Residual"); ax[0].set_title("(a) Residuals vs fitted, M2")
cvp = cv[["rmse", "mae"]]; xs = np.arange(len(cvp)); w = .38
ax[1].bar(xs - w / 2, cvp.rmse, w, yerr=cv_sd.rmse, label="RMSE", color="#1f6f8b", capsize=3)
ax[1].bar(xs + w / 2, cvp.mae, w, yerr=cv_sd.mae, label="MAE", color="#c08a1e", capsize=3)
ax[1].set_xticks(xs, cvp.index); ax[1].set_ylabel("Error on ln B (10-fold CV x20)"); ax[1].legend(frameon=False)
ax[1].set_title("(b) Out-of-sample error")
plt.tight_layout(); plt.savefig("results/figures/fig_regression_cv.pdf"); plt.close()
json.dump(R, open("results/tables/reg_stats.json", "w"), indent=1, default=float)
print(cv.round(3)); print(cv_sd.round(3))
print(json.dumps({k: R[k] for k in ("n_multi", "counts_D", "counts_A", "n_codig", "m1", "m2", "resid_sd_dairy")}, indent=1, default=float))
