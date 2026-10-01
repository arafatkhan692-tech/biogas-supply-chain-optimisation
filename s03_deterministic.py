"""Step 3 - Deterministic case study: optimum, observed-configuration comparison, validation, sensitivity."""
import sys, os, json, copy; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import instance as ins
from instance import build, P, HUB, ll_to_km, CITY, rho_from, tph
from model import build_model, solve, summary
from baselines import config_A, config_B
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
op = pd.read_pickle("data/processed/op.pkl"); P0 = copy.deepcopy(P)

def run(overrides=None, N=None, cand_rule=True):
    P.clear(); P.update(copy.deepcopy(P0))
    D = build(op, overrides=overrides, cand_rule=cand_rule)
    if N: D["N"] = N
    m = build_model(D); r = solve(m, gap=0.001); s = summary(m, D); s.update(r)
    return D, m, s, config_A(D), config_B(D)

R = {}
D, m, base, A, B = run()
_, _, base_cl, _, _ = run(N=["CL"])
Dfull, _, base_full, _, _ = run(cand_rule=False)
R.update(r_disp=D["r_disp"], Qtot=sum(D["Q"].values()), herd=sum(D["herd"].values()), beta=D["beta"], rho=rho_from(),
         crf=ins.crf(), nJ=len(D["J"]), nI=len(D["I"]), narcs=len(D["arcs"]), nJ_full=len(Dfull["J"]), narcs_full=len(Dfull["arcs"]),
         full_obj=base_full["net"], full_time=base_full["time"], screened_time=base["time"], full_fac=[list(f) for f in base_full["facilities"]],
         mmbtu_m3=ins.mmbtu_per_m3(), tph=tph())
keep = lambda s: {k: (v if k != "facilities" else [list(f) for f in v]) for k, v in s.items() if k != "flows"}
R["C_opt"] = keep(base); R["C_optCL"] = keep(base_cl); R["A"] = A; R["B"] = B
def fac_table(s):
    rows = []
    for (j, k, n) in s["facilities"]:
        thr = sum(v for (i, jj, nn), v in s["flows"].items() if jj == j and nn == n)
        src = sorted({i for (i, jj, nn) in s["flows"] if jj == j})
        hl = [D["dist"][i, j] for i in src if i != j]
        rows.append(dict(site=j, type=D["site_type"][j], level=k, tech=n, cap=P["K_levels"][k], thr=thr, util=thr / P["K_levels"][k],
                         n_src=len(src), herd=sum(D["herd"][i] for i in src), maxhaul=max(hl) if hl else 0.0))
    return rows
R["fac_C"] = fac_table(base); R["fac_CL"] = fac_table(base_cl)
f = D["farms"]; pred = f.Dairy * P["biogas_ft3_head_day"]; rep = f.Biogas; ok = rep.notna()
R.update(val_n=int(ok.sum()), val_mape=float((np.abs(pred[ok] - rep[ok]) / rep[ok]).mean() * 100),
         val_total_rep=float(rep[ok].sum()), val_total_pred=float(pred[ok].sum()), agstar_em_total=float(f.EmRed.sum()),
         em_model=base["em"], farms_status=f.Status.value_counts().to_dict(), farms_city=f.City.value_counts().to_dict(),
         herd_min=float(f.Dairy.min()), herd_max=float(f.Dairy.max()), n_ge5000=int((f.Dairy >= 5000).sum()),
         yr_op=f["Year Operational"].describe().to_dict())
f.to_csv("results/tables/maas_calgren_farms.csv", index=False)
# Fig: network maps
def draw(ax, s, title, mode):
    hub = ll_to_km(*HUB[1])
    for c, ll in CITY.items():
        x, y = ll_to_km(*ll); ax.annotate(c, (x, y), fontsize=7, color="0.45", xytext=(3, 3), textcoords="offset points")
    xy = np.array([D["coord"][i] for i in D["I"]]); h = np.array([D["herd"][i] for i in D["I"]])
    ax.scatter(xy[:, 0], xy[:, 1], s=h / 250, color="#1f6f8b", alpha=.55, zorder=2, label="Dairy (size ~ herd)")
    if mode == "B":
        pts = np.vstack([hub] + list(xy)); n = len(pts); inT = np.zeros(n, bool); inT[0] = True
        d = np.linalg.norm(pts - pts[0], axis=1); par = np.zeros(n, int)
        for _ in range(n - 1):
            d2 = np.where(inT, np.inf, d); k = int(np.argmin(d2)); inT[k] = True
            ax.plot([pts[k, 0], pts[par[k], 0]], [pts[k, 1], pts[par[k], 1]], color="#c08a1e", lw=1.4, zorder=1)
            dn = np.linalg.norm(pts - pts[k], axis=1); upd = (dn < d) & ~inT; par[upd] = k; d = np.where(upd, dn, d)
        ax.scatter(*hub, marker="*", s=180, color="#8b1f3a", zorder=3, label="Shared upgrading hub")
        ax.plot([], [], color="#c08a1e", label="Biogas gathering pipeline")
    else:
        for (i, j, n), v in s["flows"].items():
            if i != j:
                a, b = D["coord"][i], D["coord"][j]; ax.plot([a[0], b[0]], [a[1], b[1]], color="0.35", lw=0.4 + 2.5 * v / 3e5, zorder=1)
        for (j, k, n) in s["facilities"]:
            ax.scatter(*D["coord"][j], marker="s", s=60 + P["K_levels"][k] / 8000, facecolor="none", edgecolor="#8b1f3a", lw=1.6, zorder=3)
        ax.scatter([], [], marker="s", facecolor="none", edgecolor="#8b1f3a", label="Shared digester + upgrading")
        ax.plot([], [], color="0.35", label="Manure haul (width ~ t/yr)")
    ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("km (east)"); ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 4.6), sharey=True)
draw(ax[0], None, "(a) Observed hub-and-spoke (B)", "B"); draw(ax[1], base, "(b) MILP optimum (C)", "C"); ax[0].set_ylabel("km (north)")
plt.tight_layout(); plt.savefig("results/figures/fig_network_maps.pdf"); plt.close()
cfg = {"A: stand-alone\n(21 units)": A, "B: observed\nhub-and-spoke": B, "C$_{CL}$: MILP,\nCL only": base_cl, "C: MILP,\nCL or CM": base}
fig, ax = plt.subplots(figsize=(6.4, 3.0)); xs = np.arange(len(cfg)); w = .2
cols = {"capital": "#1f6f8b", "transport": "#c08a1e", "operating": "#6b8e23", "revenue": "#8b1f3a"}
for k_, (key, c) in enumerate(cols.items()):
    ax.bar(xs + (k_ - 1.5) * w, [v[key] / 1e6 * (-1 if key == "revenue" else 1) for v in cfg.values()], w, color=c,
           label=key.capitalize() + (" (negative)" if key == "revenue" else ""))
ax.plot(xs, [v["net"] / 1e6 for v in cfg.values()], "kD", ms=6, label="Net annual cost")
for x_, v in zip(xs, cfg.values()): ax.annotate(f"{v['net']/1e6:.1f}", (x_, v["net"] / 1e6), xytext=(8, -3), textcoords="offset points", fontsize=7)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(xs, list(cfg)); ax.set_ylabel("US\\$ million / yr"); ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="lower left")
plt.tight_layout(); plt.savefig("results/figures/fig_cost_comparison.pdf"); plt.close()
json.dump(R, open("results/tables/det_results.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in ("C_opt", "C_optCL", "A", "B", "fac_C", "fac_CL", "full_obj", "full_time", "screened_time", "nJ_full", "narcs_full", "val_n", "val_mape", "val_total_rep", "val_total_pred", "agstar_em_total", "farms_status", "farms_city", "herd_min", "herd_max", "n_ge5000")}, indent=1, default=float), flush=True)
sweeps = {"haul_rate": [0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00, 1.50], "lcfs_price": [0, 30, 65, 100, 150],
          "cm_yield_ratio": [1.00, 1.10, 1.20, 1.30, 1.50, 1.68], "interconnect_farm": [1.0e6, 2.0e6, 3.0e6, 5.0e6]}
SV = {}
for par, vals in sweeps.items():
    SV[par] = []
    for v in vals:
        ov = {"interconnect": dict(P0["interconnect"], farm=v)} if par == "interconnect_farm" else {par: v}
        Dx, mx, sx, Ax, Bx = run(ov)
        SV[par].append(dict(v=v, net=sx["net"], n_fac=sx["n_fac"], share=sx["share"], mean_haul=sx["mean_haul_km"], cm=sx["tech_share"]["CM"],
                            A=Ax["net"], B=Bx["net"], fac=[list(q) for q in sx["facilities"]]))
        print(par, v, round(sx["net"] / 1e6, 2), sx["n_fac"], round(sx["share"], 3), round(sx["mean_haul_km"], 1), round(sx["tech_share"]["CM"], 2),
              round(Ax["net"] / 1e6, 2), round(Bx["net"] / 1e6, 2), flush=True)
json.dump(SV, open("results/tables/det_sweeps.json", "w"), indent=1, default=float)
TOR = []
for par, lo, hi in [("haul_rate", .175, .325), ("lcfs_price", 45.5, 84.5), ("rin_price", 1.82, 3.38), ("biogas_ft3_head_day", 35, 65),
                    ("producer_share", .42, .78), ("crf_rate", .049, .091), ("upg_capex_ref", 2.1e6, 3.9e6)]:
    row = [par] + [run({par: v})[2]["net"] for v in (lo, hi)]; TOR.append(row); print(row, flush=True)
row = ["dig_capex_ref"] + [run({"dig_capex_ref": {k: x * fct for k, x in P0["dig_capex_ref"].items()}})[2]["net"] for fct in (.7, 1.3)]
TOR.append(row); print(row, flush=True)
json.dump(dict(base=base["net"], rows=TOR), open("results/tables/det_tornado.json", "w"), indent=1, default=float)
# Figures: sweeps + tornado
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
hr = SV["haul_rate"]; x = [r["v"] for r in hr]
ax[0].plot(x, [r["net"] / 1e6 for r in hr], "o-", color="#1f6f8b", label="C (MILP)"); ax[0].plot(x, [r["B"] / 1e6 for r in hr], "s--", color="#c08a1e", label="B (observed)")
ax[0].set_xlabel("Haulage rate (\\$/t-km)"); ax[0].set_ylabel("Net cost (US\\$M/yr)"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("(a) Transport cost")
a2 = ax[0].twinx(); a2.plot(x, [r["n_fac"] for r in hr], "k:", lw=1); a2.set_ylabel("Facilities (dotted)", fontsize=7); a2.spines["top"].set_visible(False)
lc = SV["lcfs_price"]; x = [r["v"] for r in lc]
ax[1].plot(x, [r["net"] / 1e6 for r in lc], "o-", color="#1f6f8b"); ax[1].plot(x, [r["B"] / 1e6 for r in lc], "s--", color="#c08a1e"); ax[1].plot(x, [r["A"] / 1e6 for r in lc], "^:", color="#8b1f3a", label="A (stand-alone)")
ax[1].axhline(0, color="k", lw=.5); ax[1].set_xlabel("LCFS credit (\\$/t CO$_2$e)"); ax[1].set_title("(b) Carbon-credit price"); ax[1].legend(frameon=False, fontsize=7)
cm = SV["cm_yield_ratio"]; x = [r["v"] for r in cm]
ax[2].plot(x, [r["net"] / 1e6 for r in cm], "o-", color="#1f6f8b"); ax[2].set_xlabel("CM/CL yield ratio"); ax[2].set_title("(c) Technology yield")
a3 = ax[2].twinx(); a3.plot(x, [r["cm"] for r in cm], "k:", lw=1); a3.set_ylabel("CM share (dotted)", fontsize=7); a3.set_ylim(-.05, 1.05); a3.spines["top"].set_visible(False)
plt.tight_layout(); plt.savefig("results/figures/fig_sensitivity.pdf"); plt.close()
lab = {"haul_rate": "Haulage rate", "lcfs_price": "LCFS credit price", "rin_price": "D3 RIN price", "biogas_ft3_head_day": "Biogas per head",
       "producer_share": "Producer revenue share", "crf_rate": "Discount rate", "upg_capex_ref": "Upgrading capex", "dig_capex_ref": "Digester capex"}
TOR.sort(key=lambda r: abs(r[2] - r[1])); b0 = base["net"] / 1e6
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for k_, (p, lo, hi) in enumerate(TOR):
    ax.barh(k_, lo / 1e6 - b0, left=b0, color="#1f6f8b", label="-30%" if k_ == 0 else None)
    ax.barh(k_, hi / 1e6 - b0, left=b0, color="#c08a1e", label="+30%" if k_ == 0 else None)
ax.axvline(b0, color="k", lw=.8); ax.set_yticks(range(len(TOR)), [lab[r[0]] for r in TOR]); ax.set_xlabel("Net annual cost of optimal network (US\\$M/yr)")
ax.legend(frameon=False, fontsize=7); plt.tight_layout(); plt.savefig("results/figures/fig_tornado.pdf"); plt.close()
print("DONE")
