"""Step 4 - Stochastic analysis in restartable phases (results checkpointed to results/tables/stoch_ckpt.json).
usage: python src/s04_stochastic.py rp | ws | cvar <w> | oos <name> | final"""
import sys, json, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from pyomo.environ import value
from instance import build
from model import build_model, solve, ysol, scen_costs
from scenarios import make, mean_scenario
from baselines import config_A, config_B
CK = "results/tables/stoch_ckpt.json"; ALPHA = 0.90
op = pd.read_pickle("data/processed/op.pkl"); D = build(op); scen = make(D, 20)
R = json.load(open(CK)) if os.path.exists(CK) else {}
save = lambda: json.dump(R, open(CK, "w"), indent=1, default=float)
enc = lambda y: sorted([list(k) for k in y]); dec = lambda L: {tuple(k): 1 for k in L}
def cvar(c, a=ALPHA):
    c = np.asarray(c); return float(min(e + np.mean(np.maximum(c - e, 0)) / (1 - a) for e in c))
def evaluate(y, sc):
    m = build_model(D, sc, w=0, fix_y=y); solve(m); return scen_costs(m)
ph = sys.argv[1]; t = time.time()
if ph == "rp":
    m = build_model(D, mean_scenario(D, scen)); solve(m, gap=0.001); yEV = ysol(m)
    R["EV"] = value(m.obj); R["y_EV"] = enc(yEV); cE = evaluate(yEV, scen); R["EEV"] = float(np.mean(cE)); R["c_EEV"] = cE
    m = build_model(D, scen, w=0); r = solve(m, gap=0.005, tlim=280); R["RP_solver"] = float(np.mean(scen_costs(m)))
    R["RP_time"] = r["time"]; R["y_RPsolver"] = enc(ysol(m)); R["c_RPsolver"] = scen_costs(m)
    R["scen"] = [dict(rho=s["rho_val"], theta=s["theta"], phi=s["phi"], lcfs=s["lcfs"], rin=s["rin"], gas=s["gas"], shock=s["shock"],
                      Qsum=sum(s["Q"].values())) for s in scen]
elif ph == "ws":
    ws = []
    for s in scen:
        mw = build_model(D, [dict(s, p=1.0)]); solve(mw, gap=0.001, tlim=60); ws.append(value(mw.obj))
    R["ws"] = ws; R["WS"] = float(np.mean(ws))
elif ph == "cvar":
    w = float(sys.argv[2]); mw = build_model(D, scen, w=w, alpha=ALPHA); r = solve(mw, gap=0.005, tlim=230); c = scen_costs(mw)
    R.setdefault("front", {})[str(w)] = dict(w=w, E=float(np.mean(c)), CVaR=cvar(c), y=enc(ysol(mw)), time=r["time"], c=c)
elif ph == "oos":
    oos = make(D, 200, seed=99); name = sys.argv[2]; y = dec(R["designs"][name]); c = []
    for b in range(0, 200, 50): c += evaluate(y, [dict(s, p=1 / 50) for s in oos[b:b + 50]])
    R.setdefault("oos_c", {})[name] = c
elif ph == "base_oos":
    oos = make(D, 200, seed=99)
    R.setdefault("oos_c", {})["A: stand-alone"] = [config_A(D, s)["net"] for s in oos]
    R["oos_c"]["B: observed hub"] = [config_B(D, s)["net"] for s in oos]
    R["oos_shock"] = float(np.mean([s["shock"] for s in oos]))
    # in-sample scores of A, B too
    R["ins_A"] = [config_A(D, s)["net"] for s in scen]; R["ins_B"] = [config_B(D, s)["net"] for s in scen]
R.setdefault("log", []).append([ph] + sys.argv[2:] + [time.time() - t]); save()
print(ph, sys.argv[2:], round(time.time() - t, 1), {k: (v if not isinstance(v, list) or len(v) < 6 else "...") for k, v in R.items() if k in ("EV", "EEV", "RP_solver", "RP_time", "y_EV", "y_RPsolver", "WS")})
