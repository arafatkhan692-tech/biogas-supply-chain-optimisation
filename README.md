# Optimisation of a Livestock-Based Biogas Supply Chain Network

MATH6004 Industrial Engineering Masters Project, Curtin University, Semester 2 2026
Student: Md Yeaseer Arafat Khan (22950886) · Supervisor: Sam Wiwatanapataphee

This repository contains the data processing, modelling and analysis code for the project. It uses the
EPA AgSTAR Livestock Anaerobic Digester Database to (1) characterise U.S. livestock digester projects,
(2) estimate biogas yield from herd size and technology, (3) design a shared digester network for the
Maas Calgren dairy cluster (Tulare and Kern Counties, California) with a mixed-integer linear programme (MILP),
and (4) extend the design to a two-stage stochastic MILP with a Conditional Value-at-Risk (CVaR) term.

## Repository structure

```
data/raw/            AgSTAR Livestock Anaerobic Digester Database (xlsx, as downloaded from EPA)
data/processed/      cleaned AgSTAR tables (CSV) produced by step 1
notebooks/           Colab notebook that runs the whole pipeline and shows the figures
src/                 Python code (steps s01-s05 plus four shared modules)
results/figures/     all figures used in the report (PDF)
results/tables/      numeric results (JSON / CSV) behind every table and number in the report
run_all.sh           runs the whole pipeline in order
requirements.txt     Python packages
```

### Code in `src/`

| File | Purpose | Report section |
|---|---|---|
| `s01_preprocess_eda.py` | Cleans the AgSTAR sheets (state/county names, numeric fields), saves clean CSVs, produces the EDA statistics and figures, lists the Californian clusters | 5.2-5.3, 6.1, 6.3 |
| `s02_regression.py` | Log-log regression of biogas on herd size (M1) and with technology, species and co-digestion (M2); 10-fold x 20 cross-validation against a Monte Carlo baseline; saves residuals for the yield scenarios | 5.4.1-5.4.2, 6.2 |
| `instance.py` | Builds the Maas Calgren case-study instance: farms, synthetic locations calibrated to the 22-mile pipeline, candidate sites, distances, costs, yields and revenue. **All parameters are in the `PARAMS` dictionary at the top.** | 5.5 |
| `model.py` | Pyomo MILP. One scenario with probability 1 = deterministic model (Sec. 5.4.3); several scenarios + weight `w` = two-stage stochastic model with CVaR (Sec. 5.4.4) | 5.4.3-5.4.4 |
| `baselines.py` | Scores the observed configurations: A (stand-alone digesters) and B (observed hub-and-spoke with biogas pipeline) | 5.5, 6.4 |
| `scenarios.py` | Monte Carlo scenarios for prices (gas, RIN, LCFS with policy shock), yield, manure supply and haulage cost | 5.4.4 |
| `s03_deterministic.py` | Optimal network, comparison with A and B, validation against AgSTAR, sensitivity sweeps and tornado chart | 6.4-6.5 |
| `s04_stochastic.py` | Stochastic analysis in restartable phases: `rp`, `ws`, `cvar <w>`, `oos <design>`, `base_oos` | 6.6 |
| `s05_stochastic_figures.py` | Risk-cost frontier and out-of-sample figure | 6.6 |

## How to run

Python 3.10 or later.

```bash
pip install -r requirements.txt
bash run_all.sh                      # full pipeline, about 20-25 minutes on one core
```

Each step can also be run on its own from the repository root, for example `python src/s03_deterministic.py`.
Steps 1-3 take about 3 minutes in total. The solver is HiGHS (installed with `highspy`); no licence is needed.
The deterministic model solves in about 4 seconds and the 20-scenario stochastic model in about 2.5 minutes.

`results/tables/stoch_ckpt.json` already holds the stochastic results used in the report, so
`python src/s05_stochastic_figures.py` redraws the stochastic figure without re-solving anything.

To run in Google Colab, open `notebooks/run_pipeline.ipynb`.

## Key results (base case, US$ million per year; negative = net surplus)

| Design | Net annual cost |
|---|---|
| A: each dairy builds its own digester and upgrading | -15.64 |
| B: observed hub-and-spoke (pipeline to Pixley hub) | -24.70 |
| C_CL: MILP optimum, covered lagoons only | -24.45 |
| C: MILP optimum, covered lagoon or complete mix | -29.42 |

Stochastic analysis (20 scenarios): VSS is about zero (within the 0.5% solver gap), EVPI = 1.03, and the
CVaR(0.9) of the expected-value design is a loss of 6.95. Increasing the CVaR weight shifts the design from
complete-mix to covered-lagoon digesters.

## Assumptions to be aware of

- AgSTAR gives only the city and county of each project. Farm locations are therefore **synthetic**: sampled
  around town centroids (fixed seed) and calibrated so the gathering pipeline length matches the reported 22 miles.
- Capital, operating, interconnection and haulage costs are engineering assumptions, marked `ASSUMPTION` in
  `instance.py`. Their effect is tested in the sensitivity and stochastic analyses.
- AgSTAR's biogas (50 ft3/head/day for 44% of dairy projects) and emission-reduction fields appear to be
  estimates, so validation against them checks consistency rather than metered performance.

## Data source

U.S. EPA AgSTAR Program. Livestock Anaerobic Digester Database.
https://www.epa.gov/agstar/livestock-anaerobic-digester-database
