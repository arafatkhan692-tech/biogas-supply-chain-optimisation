"""Module - Observed configurations scored with the same parameters as the MILP.
A: stand-alone - every dairy builds its own covered lagoon digester + upgrading + interconnection.
B: observed hub-and-spoke - on-farm covered lagoons + blowers, MST gathering pipeline (x circuity)
   to one shared upgrading/injection hub at Pixley (the publicly described Calgren design).
Both can be scored under any scenario dict (Q, theta, rho, phi)."""
import numpy as np
from instance import P, crf, fac_capex, mst_length, HUB, ll_to_km


def smallest_level(q):
    for k, c in P["K_levels"].items():
        if q <= c: return k, c
    return "XL", P["K_levels"]["XL"]


def _rho(sc):  # FIX: single scalar price; farm ids are not all keys of sc["rho"] after site screening
    return float(np.mean(list(sc["rho"].values())))


def config_A(D, sc=None):
    sc = sc or dict(Q=D["Q"], theta=1.0, rho=D["rho"], phi=1.0)
    a = crf(); cap = sum(a * fac_capex(smallest_level(D["Q"][i])[0], "CL", "farm") for i in D["I"])
    q = sum(sc["Q"].values()); op = q * P["cop"]["CL"]; bio = q * D["beta"]["CL"] * sc["theta"]; rev = bio * _rho(sc)
    return dict(n_fac=len(D["I"]), capital=cap, transport=0.0, operating=op, revenue=rev, net=cap + op - rev,
                biogas=bio, pipe_km=0.0, tkm=0.0)


def config_B(D, sc=None):
    sc = sc or dict(Q=D["Q"], theta=1.0, rho=D["rho"], phi=1.0)
    capex = 0.0
    for i in D["I"]:
        k, c = smallest_level(D["Q"][i]); capex += P["dig_capex_ref"]["CL"] * (c / P["K_ref"]) ** P["dig_exp"] + P["blower_capex"]
    hub = np.array(ll_to_km(*HUB[1])); pts = np.vstack([hub] + [np.array(D["coord"][i]) for i in D["I"]])
    pipe = mst_length(pts) * P["circuity"]; capex += pipe * P["pipe_capex_per_km"]
    capex += P["upg_capex_ref"] * (sum(D["Q"].values()) / P["K_ref"]) ** P["upg_exp"] + P["interconnect"]["hub"]
    cap = crf() * capex
    q = sum(sc["Q"].values()); op = q * P["cop"]["CL"]; bio = q * D["beta"]["CL"] * sc["theta"]; rev = bio * _rho(sc)
    return dict(n_fac=len(D["I"]) + 1, capital=cap, transport=0.0, operating=op, revenue=rev, net=cap + op - rev,
                biogas=bio, pipe_km=pipe, tkm=0.0)
