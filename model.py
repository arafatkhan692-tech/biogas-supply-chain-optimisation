"""Module - MILP models (Pyomo). Deterministic = Section 5.4.3; two-stage stochastic with CVaR = 5.4.4.
A deterministic run is one scenario with probability 1 and w = 0."""
import pyomo.environ as pyo
import time


def get_solver(gap=0.005, tlim=600):
    s = pyo.SolverFactory("appsi_highs"); s.config.mip_gap = gap; s.config.time_limit = tlim
    s.config.load_solution = True
    return "HiGHS", s


def base_scenario(D):
    return [dict(p=1.0, Q=D["Q"], theta=1.0, rho=D["rho"], phi=1.0)]


def build_model(D, scen=None, w=0.0, alpha=0.90, fix_y=None):
    from instance import P
    scen = scen or base_scenario(D)
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=D["I"]); m.J = pyo.Set(initialize=D["J"])
    m.K = pyo.Set(initialize=D["K"]); m.N = pyo.Set(initialize=D["N"])
    m.S = pyo.Set(initialize=range(len(scen))); m.A = pyo.Set(dimen=2, initialize=D["arcs"])
    m.y = pyo.Var(m.J, m.K, m.N, domain=pyo.Binary)
    m.x = pyo.Var(m.A, m.N, m.S, domain=pyo.NonNegativeReals)

    def ct(i, j, s):
        return 0.0 if i == j else P["load_cost"] + P["haul_rate"] * scen[s]["phi"] * D["dist"][i, j]
    m.capital = pyo.Expression(expr=sum(D["ccap"][j, k, n] * m.y[j, k, n] for j in D["J"] for k in D["K"] for n in D["N"]))

    def stage2(mm, s):
        sc = scen[s]
        return sum((ct(i, j, s) + D["cop"][j, n] - sc["rho"][j] * D["beta"][n] * sc["theta"]) * mm.x[i, j, n, s]
                   for (i, j) in D["arcs"] for n in D["N"])
    m.C2 = pyo.Expression(m.S, rule=stage2)
    m.Ctot = pyo.Expression(m.S, rule=lambda mm, s: mm.capital + mm.C2[s])
    p = [sc["p"] for sc in scen]
    m.Exp = pyo.Expression(expr=sum(p[s] * m.Ctot[s] for s in m.S))
    if w > 0:
        m.eta = pyo.Var(domain=pyo.Reals); m.u = pyo.Var(m.S, domain=pyo.NonNegativeReals)
        m.cvar_def = pyo.Constraint(m.S, rule=lambda mm, s: mm.u[s] >= mm.Ctot[s] - mm.eta)
        m.CVaR = pyo.Expression(expr=m.eta + sum(p[s] * m.u[s] for s in m.S) / (1 - alpha))
        m.obj = pyo.Objective(expr=(1 - w) * m.Exp + w * m.CVaR)
    else:
        m.obj = pyo.Objective(expr=m.Exp)
    out, inn = {}, {}
    for (i, j) in D["arcs"]: out.setdefault(i, []).append(j); inn.setdefault(j, []).append(i)
    m.supply = pyo.Constraint(m.I, m.S, rule=lambda mm, i, s: sum(mm.x[i, j, n, s] for j in out[i] for n in mm.N) <= scen[s]["Q"][i])
    m.capacity = pyo.Constraint(m.J, m.N, m.S, rule=lambda mm, j, n, s:
                                sum(mm.x[i, j, n, s] for i in inn.get(j, [])) <= sum(D["Kcap"][j, k] * mm.y[j, k, n] for k in mm.K)
                                if j in inn else pyo.Constraint.Skip)
    m.single = pyo.Constraint(m.J, rule=lambda mm, j: sum(mm.y[j, k, n] for k in mm.K for n in mm.N) <= 1)
    # valid inequality: flow only into an installed technology (tightens LP relaxation)
    m.strong = pyo.Constraint(m.A, m.N, m.S, rule=lambda mm, i, j, n, s:
                              mm.x[i, j, n, s] <= scen[s]["Q"][i] * sum(mm.y[j, k, n] for k in mm.K))
    if fix_y is not None:
        for key in m.y: m.y[key].fix(1 if fix_y.get(key, 0) > 0.5 else 0)
    return m


def solve(m, gap=0.005, tlim=600):
    name, s = get_solver(gap, tlim)
    t = time.time(); r = s.solve(m); dt = time.time() - t
    try: bound = r.best_objective_bound
    except Exception: bound = None
    return dict(solver=name, time=dt, obj=pyo.value(m.obj), bound=bound)


def ysol(m):
    return {k: 1 for k in m.y if pyo.value(m.y[k]) > 0.5}


def scen_costs(m):
    return [pyo.value(m.Ctot[s]) for s in m.S]


def summary(m, D, s=0):
    from instance import P, tph
    fac = list(ysol(m))
    flows = {(i, j, n): pyo.value(m.x[i, j, n, s]) for (i, j) in D["arcs"] for n in D["N"] if pyo.value(m.x[i, j, n, s]) > 1e-3}
    treated = sum(flows.values()); Qtot = sum(D["Q"].values())
    tkm = sum(v * D["dist"][i, j] for (i, j, n), v in flows.items() if i != j)
    hauled = sum(v for (i, j, n), v in flows.items() if i != j)
    trans = sum(v * (0 if i == j else P["load_cost"] + P["haul_rate"] * D["dist"][i, j]) for (i, j, n), v in flows.items())
    opx = sum(v * D["cop"][j, n] for (i, j, n), v in flows.items())
    rev = sum(v * D["rho"][j] * D["beta"][n] for (i, j, n), v in flows.items())
    biogas = sum(v * D["beta"][n] for (i, j, n), v in flows.items())
    cap = sum(D["ccap"][f] for f in fac)
    em = treated / tph() * P["em_t_per_head"] - tkm * P["truck_kgco2_tkm"] / 1000
    return dict(facilities=fac, n_fac=len(fac), flows=flows, treated=treated, share=treated / Qtot, tkm=tkm, hauled=hauled,
                mean_haul_km=(tkm / hauled if hauled else 0.0), capital=cap, transport=trans, operating=opx, revenue=rev,
                net=cap + trans + opx - rev, biogas=biogas, em=em, truck_em=tkm * P["truck_kgco2_tkm"] / 1000,
                tech_share={n: sum(v for (i, j, nn), v in flows.items() if nn == n) / max(treated, 1) for n in D["N"]})
