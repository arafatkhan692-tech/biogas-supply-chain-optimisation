"""Module - Scenario generation (Section 5.4.4).
rho_s : RNG netback from LCFS (lognormal median $65/t, log-sd 0.40), D3 RIN (median $2.60, log-sd 0.30) and
        gas price (median $3.0/MMBtu, log-sd 0.30); with prob. 0.10 the LCFS credit is lost (policy shock).
theta_s : cluster yield multiplier = exp(mean of 21 resampled dairy regression residuals), re-centred to E=1.
Q_is : manure availability per dairy, N(1, 0.10) truncated to [0.75, 1.20].
phi_s : haulage-rate multiplier, lognormal log-sd 0.15."""
import numpy as np
from instance import rho_from


def make(D, S, seed=2026, p_shock=0.10, resid=None):
    rng = np.random.default_rng(seed)
    if resid is None: resid = np.load("results/tables/resid_m1.npy")
    nI = len(D["I"]); corr = np.array([np.exp(rng.choice(resid, nI).mean()) for _ in range(20000)]).mean()
    out = []
    for s in range(S):
        lcfs = 65 * np.exp(rng.normal(0, .40)); rin = 2.6 * np.exp(rng.normal(0, .30)); gas = 3.0 * np.exp(rng.normal(0, .30))
        shock = rng.uniform() < p_shock
        if shock: lcfs = 0.0
        r = rho_from(gas, rin, lcfs); theta = np.exp(rng.choice(resid, nI).mean()) / corr
        q = {i: D["Q"][i] * float(np.clip(rng.normal(1, .10), .75, 1.20)) for i in D["I"]}
        out.append(dict(p=1.0 / S, Q=q, theta=float(theta), rho={j: r for j in D["J"]}, phi=float(np.exp(rng.normal(0, .15))),
                        lcfs=lcfs, rin=rin, gas=gas, rho_val=r, shock=bool(shock)))
    return out


def mean_scenario(D, scen):
    q = {i: float(np.mean([sc["Q"][i] for sc in scen])) for i in D["I"]}; r = float(np.mean([sc["rho_val"] for sc in scen]))
    return [dict(p=1.0, Q=q, theta=float(np.mean([sc["theta"] for sc in scen])), rho={j: r for j in D["J"]},
                 phi=float(np.mean([sc["phi"] for sc in scen])), rho_val=r)]
