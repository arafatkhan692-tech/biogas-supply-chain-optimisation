"""Module - Case-study instance: Maas Calgren dairy digester cluster (Tulare & Kern Counties, CA).

Supply nodes are the 21 AgSTAR projects in the 'Maas Calgren' cluster. AgSTAR gives only city/county,
so each dairy is placed at a synthetic location inside a disc of radius r_disp km around its city
centroid (seeded). r_disp is calibrated so that the minimum spanning tree linking the 12 dairies
nearest the Pixley hub (x circuity) is close to the ~35 km (22 mile) gathering pipeline reported
publicly for the cluster. All money in 2024 US$. Values marked ASSUMPTION are engineering estimates
that are stress-tested in the sensitivity and stochastic analyses.
"""
import numpy as np, math

CITY = {"Pixley": (35.9686, -119.2918), "Tipton": (36.0594, -119.3121), "Tulare": (36.2077, -119.3473),
        "Porterville": (36.0652, -119.0168), "Wasco": (35.5941, -119.3409)}
HUB = ("HUB_Pixley", (35.9550, -119.3000))

PARAMS = dict(
    manure_kg_head_day=68.0,      # ASABE D384.2, lactating Holstein, as excreted
    recoverable=0.85,             # ASSUMPTION
    biogas_ft3_head_day=50.0,     # AgSTAR modal value (Section 6.1)
    cm_yield_ratio=1.30,          # ASSUMPTION, conservative vs regression M2
    crf_rate=0.07, crf_years=20,
    K_levels={"S": 75_000, "M": 200_000, "L": 500_000, "XL": 1_200_000},
    K_ref=100_000,
    dig_capex_ref={"CL": 3.5e6, "CM": 7.0e6}, dig_exp=0.70,     # ASSUMPTION
    upg_capex_ref=3.0e6, upg_exp=0.60,                           # ASSUMPTION
    interconnect={"farm": 3.0e6, "town": 2.0e6, "hub": 0.75e6},  # ASSUMPTION
    cop={"CL": 3.0, "CM": 4.7},   # $/t incl. upgrading O&M, ASSUMPTION
    load_cost=2.5, haul_rate=0.25, circuity=1.3,
    ch4_frac=0.60, upg_recovery=0.97, gas_price=3.0, rin_price=2.60, rin_per_mmbtu=11.727,
    lcfs_price=65.0, lcfs_t_per_mmbtu=0.38, producer_share=0.60,
    pipe_capex_per_km=0.25e6, blower_capex=0.30e6,
    truck_kgco2_tkm=0.10, em_t_per_head=8.89,
    max_haul_km=35.0,
    cand_herd_min=5000,
)
P = PARAMS


def crf(r=None, n=None):
    r = P["crf_rate"] if r is None else r; n = P["crf_years"] if n is None else n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def mmbtu_per_m3():
    return P["ch4_frac"] * 35.3147 * 1012 / 1e6 * P["upg_recovery"]


def rho_from(gas=None, rin=None, lcfs=None, share=None):
    gas = P["gas_price"] if gas is None else gas; rin = P["rin_price"] if rin is None else rin
    lcfs = P["lcfs_price"] if lcfs is None else lcfs; share = P["producer_share"] if share is None else share
    usd_mmbtu = gas + rin * P["rin_per_mmbtu"] + lcfs * P["lcfs_t_per_mmbtu"]
    return share * usd_mmbtu * mmbtu_per_m3()


def tph():
    return P["manure_kg_head_day"] / 1000 * 365 * P["recoverable"]


def beta():
    bcl = P["biogas_ft3_head_day"] * 0.0283168 * 365 / tph()
    return {"CL": bcl, "CM": bcl * P["cm_yield_ratio"]}


def fac_capex(k, n, stype):
    s = P["K_levels"][k] / P["K_ref"]
    return P["dig_capex_ref"][n] * s ** P["dig_exp"] + P["upg_capex_ref"] * s ** P["upg_exp"] + P["interconnect"][stype]


def ll_to_km(lat, lon, lat0=35.95, lon0=-119.30):
    return ((lon - lon0) * 111.32 * math.cos(math.radians(lat0)), (lat - lat0) * 110.57)


def mst_length(pts):
    pts = np.asarray(pts); n = len(pts); inT = np.zeros(n, bool); inT[0] = True
    d = np.linalg.norm(pts - pts[0], axis=1); tot = 0.0
    for _ in range(n - 1):
        d2 = np.where(inT, np.inf, d); k = int(np.argmin(d2)); tot += d2[k]; inT[k] = True
        d = np.minimum(d, np.linalg.norm(pts - pts[k], axis=1))
    return tot


def farms(op, cluster="Maas Calgren"):
    f = op[op["Cluster Name"] == cluster][["Project Name", "City", "County", "Dairy", "Biogas", "Status",
                                            "Digester Type", "EmRed", "Year Operational"]].copy()
    f["id"] = ["F%02d" % (i + 1) for i in range(len(f))]
    return f.reset_index(drop=True)


def place(f, r_disp, seed=11):
    rng = np.random.default_rng(seed); xy = []
    for c in f.City:
        cx, cy = ll_to_km(*CITY[c]); a = rng.uniform(0, 2 * np.pi); r = r_disp * np.sqrt(rng.uniform())
        xy.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(xy)


def calibrate_dispersion(f, target_km=35.4, seed=11):
    hub = np.array(ll_to_km(*HUB[1])); best = None
    for r in np.arange(1.0, 12.01, 0.25):
        xy = place(f, r, seed); d = np.linalg.norm(xy - hub, axis=1); idx = np.argsort(d)[:12]
        L = mst_length(np.vstack([hub, xy[idx]])) * P["circuity"]
        if best is None or abs(L - target_km) < abs(best[1] - target_km): best = (r, L)
    return best


def build(op, r_disp=None, seed=11, overrides=None, cand_rule=True):
    if overrides: P.update(overrides)
    f = farms(op)
    if r_disp is None: r_disp, _ = calibrate_dispersion(f, seed=seed)
    xy = place(f, r_disp, seed)
    I = list(f.id); herd = dict(zip(I, f.Dairy))
    Q = {i: herd[i] * tph() for i in I}
    thr = P["cand_herd_min"] if cand_rule else 0
    sites = {i: (tuple(xy[k]), "farm") for k, i in enumerate(I) if herd[i] >= thr}
    sites[HUB[0]] = (ll_to_km(*HUB[1]), "hub")
    for c in ["Tipton", "Tulare", "Porterville", "Wasco"]:
        sites["T_" + c] = (ll_to_km(*CITY[c]), "town")
    J = list(sites)
    coord = {**{i: tuple(xy[k]) for k, i in enumerate(I)}, **{j: sites[j][0] for j in J}}
    dist = {(i, j): math.dist(coord[i], coord[j]) * P["circuity"] for i in I for j in J}
    arcs = [(i, j) for i in I for j in J if dist[i, j] <= P["max_haul_km"]]
    K = list(P["K_levels"]); N = ["CL", "CM"]; Kcap = {(j, k): P["K_levels"][k] for j in J for k in K}
    a = crf(); ccap = {(j, k, n): a * fac_capex(k, n, sites[j][1]) for j in J for k in K for n in N}
    return dict(I=I, J=J, K=K, N=N, Q=Q, herd=herd, arcs=arcs, dist=dist, Kcap=Kcap, ccap=ccap,
                cop={(j, n): P["cop"][n] for j in J for n in N}, beta=beta(), rho={j: rho_from() for j in J},
                coord=coord, site_type={j: sites[j][1] for j in J}, farms=f, r_disp=r_disp)
