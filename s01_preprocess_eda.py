"""Step 1 - Data preparation and exploratory analysis of the AgSTAR database."""
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
F = "data/raw/agstarlivestockaddatabase.xlsx"
op = pd.read_excel(F, "Operational and Construction"); sh = pd.read_excel(F, "Shutdown")
for d in (op, sh):
    d["State"] = d["State"].str.strip().str.upper()
    d["County"] = d["County"].astype(str).str.strip().str.title()
    d["Biogas"] = pd.to_numeric(d["Biogas Generation Estimate (cu-ft/day)"], errors="coerce")
    d["Elec"] = pd.to_numeric(d["Electricity Generated (kWh/yr)"], errors="coerce")
op["EmRed"] = op["Total Emission Reductions (MTCO2e/yr)"]
op.to_pickle("data/processed/op.pkl"); sh.to_pickle("data/processed/sh.pkl")
op.to_csv("data/processed/agstar_operational_clean.csv", index=False); sh.to_csv("data/processed/agstar_shutdown_clean.csv", index=False)
S = {}
S["n_op"] = len(op); S["n_sh"] = len(sh); S["status"] = op.Status.value_counts().to_dict()
S["n_states"] = op.State.nunique(); S["ptype"] = op["Project Type"].value_counts().to_dict()
S["biogas_cov"] = int(op.Biogas.notna().sum()); S["elec_cov"] = int(op.Elec.notna().sum())
S["dairy_cov"] = int(op.Dairy.notna().sum()); S["em_cov"] = int(op.EmRed.notna().sum())
S["CA_clusters"] = op[op.State == "CA"]["Cluster Name"].nunique(); S["clusters"] = op["Cluster Name"].nunique()
S["CA_n"] = int((op.State == "CA").sum())
other = ["Plug Flow - Unspecified", "Vertical Plug Flow", "Complete Mix Mini Digester",
         "Primary digester tank with secondary covered lagoon", "Fixed Film/Attached Media",
         "Anaerobic Sequencing Batch Reactor", "Induced Blanket Reactor", "Unknown or Unspecified"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
st = op.State.value_counts().head(12)[::-1]
ax[0].barh(st.index, st.values, color="#1f6f8b"); ax[0].set_xlabel("Number of projects"); ax[0].set_title("(a) Top 12 states")
dt = op["Digester Type"].replace({o: "Other / unspecified" for o in other}).value_counts()[::-1]
ax[1].barh(dt.index, dt.values, color="#c08a1e"); ax[1].set_title("(b) Digester type"); ax[1].set_xlabel("Number of projects")
plt.tight_layout(); plt.savefig("results/figures/fig_state_digester.pdf"); plt.close()
S["dtype"] = op["Digester Type"].value_counts().to_dict(); S["state_top"] = op.State.value_counts().head(6).to_dict()
eu = op["Biogas End Use(s)"].dropna().str.split(";").explode().str.strip().replace({"Pipeline to Electricity": "Electricity"})
eu = eu.value_counts(); S["enduse"] = eu.to_dict()
ani = op["Animal/Farm Type(s)"].fillna("Unknown").str.split(";").explode().str.strip().replace({"Broiler": "Poultry"}).value_counts()
S["animal"] = ani.to_dict()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
e = eu.head(7)[::-1]; ax[0].barh(e.index, e.values, color="#1f6f8b")
ax[0].set_title("(a) Biogas end use (multi-use split)"); ax[0].set_xlabel("Mentions")
isrng = lambda d: d["Biogas End Use(s)"].fillna("").str.contains("CNG|Pipeline Gas")
S["rng_share_CA"] = float(isrng(op[op.State == "CA"]).mean()); S["rng_share_nonCA"] = float(isrng(op[op.State != "CA"]).mean())
yr = op.dropna(subset=["Year Operational"]); bins = np.arange(1978, 2026, 2)
ax[1].hist([yr[yr.State == "CA"]["Year Operational"], yr[yr.State != "CA"]["Year Operational"]], bins=bins, stacked=True,
           color=["#c08a1e", "#1f6f8b"], label=["California", "Other states"])
ax[1].legend(frameon=False); ax[1].set_title("(b) Year operational"); ax[1].set_xlabel("Year"); ax[1].set_ylabel("Projects")
plt.tight_layout(); plt.savefig("results/figures/fig_enduse_year.pdf"); plt.close()
S["since2018"] = int((op["Year Operational"] >= 2018).sum())
S["CA_since2018"] = int(((op["Year Operational"] >= 2018) & (op.State == "CA")).sum())
m = (op.Biogas > 0) & (op.Dairy > 0); x = np.log(op.Dairy[m]); y = np.log(op.Biogas[m])
r = stats.linregress(x, y)
S["reg"] = dict(n=int(m.sum()), b0=r.intercept, b1=r.slope, R2=r.rvalue ** 2, se1=r.stderr, p=r.pvalue,
                ci1=[r.slope - 1.96 * r.stderr, r.slope + 1.96 * r.stderr])
ratio = op.Biogas[m] / op.Dairy[m]; is50 = np.abs(ratio - 50) < 0.5
S["ratio50_share"] = float(is50.mean()); S["ratio50_n"] = int(is50.sum()); S["ratio_median"] = float(ratio.median())
fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.scatter(op.Dairy[m][~is50], op.Biogas[m][~is50], s=12, alpha=.7, color="#1f6f8b", label="Other reported values")
ax.scatter(op.Dairy[m][is50], op.Biogas[m][is50], s=12, alpha=.7, marker="s", color="#c08a1e", label="Exactly 50 cu-ft/head/day")
xx = np.linspace(x.min(), x.max(), 50)
ax.plot(np.exp(xx), np.exp(r.intercept + r.slope * xx), "k-", lw=1.2, label=f"Fit: ln B = {r.intercept:.2f} + {r.slope:.2f} ln N")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Dairy herd size N (head)"); ax.set_ylabel("Biogas B (cu-ft/day)")
ax.legend(frameon=False, fontsize=7); plt.tight_layout(); plt.savefig("results/figures/fig_herd_biogas.pdf"); plt.close()
em = op[(op.Dairy > 0) & (op.EmRed > 0) & (op["Animal/Farm Type(s)"] == "Dairy")]; eph = em.EmRed / em.Dairy
S["em_per_head_median"] = float(eph.median()); S["em_per_head_iqr"] = [float(eph.quantile(.25)), float(eph.quantile(.75))]
ca = op[op.State == "CA"].dropna(subset=["Cluster Name"])
g = ca.groupby("Cluster Name").agg(Projects=("Project Name", "size"), Counties=("County", lambda s: ", ".join(sorted(set(s)))),
                                   Herd=("Dairy", "sum"), Digester=("Digester Type", lambda s: s.mode()[0]),
                                   EndUse=("Biogas End Use(s)", lambda s: s.mode()[0]), Em=("EmRed", "sum")).sort_values("Projects", ascending=False)
g.to_csv("results/tables/ca_clusters.csv"); S["CA_cluster_projects"] = int(g.Projects.sum())
sh["Life"] = sh["Year Shutdown"] - sh["Year Operational"]
S["sh_life_median"] = float(sh.Life.median()); S["sh_life_cov"] = int(sh.Life.notna().sum())
S["sh_dtype"] = sh["Digester Type"].value_counts().head(4).to_dict(); S["sh_state"] = sh.State.value_counts().head(5).to_dict()
iselec = lambda d: d["Biogas End Use(s)"].fillna("").str.contains("Electricity|Cogeneration")
S["elec_share_op"] = float(iselec(op).mean()); S["elec_share_sh"] = float(iselec(sh).mean())
S["rng_share_op"] = float(isrng(op).mean()); S["rng_share_sh"] = float(isrng(sh).mean())
S["sh_dairy_median"] = float(sh.Dairy.median()); S["op_dairy_median"] = float(op.Dairy.median())
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].hist(sh.Life.dropna(), bins=range(0, 32, 2), color="#8b1f3a")
ax[0].set_xlabel("Years in operation before shutdown"); ax[0].set_ylabel("Projects"); ax[0].set_title(f"(a) Shut-down projects (n={S['sh_life_cov']})")
xs = np.arange(2); w = 0.38
ax[1].bar(xs - w / 2, [S["elec_share_op"], S["rng_share_op"]], w, label="Operational/constr.", color="#1f6f8b")
ax[1].bar(xs + w / 2, [S["elec_share_sh"], S["rng_share_sh"]], w, label="Shut down", color="#8b1f3a")
ax[1].set_xticks(xs, ["Electricity/\nCHP", "RNG/CNG"]); ax[1].set_ylabel("Share of projects")
ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("(b) End-use pathway by status")
plt.tight_layout(); plt.savefig("results/figures/fig_shutdown.pdf"); plt.close()
json.dump(S, open("results/tables/eda_stats.json", "w"), indent=1, default=float)
print(json.dumps(S, indent=1, default=float)); print(g.to_string())
