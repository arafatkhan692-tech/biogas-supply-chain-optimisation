"""Step 5 - Figure for the stochastic results (risk frontier and out-of-sample box plots)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("results/tables/stoch_ckpt.json"))
def cvar(c, a=.9):
    c = np.asarray(c); return min(e + np.mean(np.maximum(c - e, 0)) / (1 - a) for e in c)
fr = [(0.0, R["EEV"], cvar(R["c_EEV"]))] + [(f["w"], f["E"], f["CVaR"]) for f in sorted(R["front"].values(), key=lambda f: f["w"])]
oos = {n: np.array(c) / 1e6 for n, c in R["oos_c"].items()}
pts = {"A": (np.mean(R["ins_A"]), cvar(R["ins_A"])), "B": (np.mean(R["ins_B"]), cvar(R["ins_B"]))}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
ax[0].plot([f[1] / 1e6 for f in fr], [f[2] / 1e6 for f in fr], "o-", color="#1f6f8b", label="Stochastic MILP, varying $w$")
for w, E, C in fr: ax[0].annotate(f"w={w:g}", (E / 1e6, C / 1e6), fontsize=7, xytext=(4, 3), textcoords="offset points")
for k, (E, C) in pts.items():
    ax[0].scatter(E / 1e6, C / 1e6, marker="s" if k == "B" else "^", color="#c08a1e" if k == "B" else "#8b1f3a", zorder=3,
                  label={"A": "A: stand-alone", "B": "B: observed hub"}[k])
ax[0].set_xlabel("Expected net cost (US\\$M/yr)"); ax[0].set_ylabel("CVaR$_{0.9}$ (US\\$M/yr)"); ax[0].legend(frameon=False, fontsize=6.5)
ax[0].set_title("(a) Risk-cost frontier (20 scenarios)")
names = ["A: stand-alone", "B: observed hub", "EV (= deterministic C)", "CVaR w=0.5", "CVaR w=0.75", "CVaR w=1"]
lab = ["A", "B", "C / RP", "w=0.5", "w=0.75", "w=1"]
ax[1].boxplot([oos[n] for n in names], whis=(5, 95), showfliers=False, widths=.55)
ax[1].scatter(range(1, 7), [cvar(oos[n]) for n in names], marker="v", color="#8b1f3a", zorder=3, label="CVaR$_{0.9}$")
ax[1].axhline(0, color="k", lw=.5); ax[1].set_xticks(range(1, 7), lab); ax[1].set_ylabel("Net cost (US\\$M/yr)")
ax[1].set_title("(b) Out-of-sample, 200 scenarios"); ax[1].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig("results/figures/fig_stochastic.pdf"); plt.close()
print(fr)
