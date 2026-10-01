#!/usr/bin/env bash
# Runs the full pipeline from the repository root. Total time on one CPU core: about 20-25 minutes,
# most of it in the stochastic step. Each stochastic phase saves to results/tables/stoch_ckpt.json,
# so the script can be stopped and restarted from any phase.
set -e
cd "$(dirname "$0")"
python src/s01_preprocess_eda.py
python src/s02_regression.py
python src/s03_deterministic.py
python src/s04_stochastic.py rp          # EV, EEV and risk-neutral recourse problem (~3 min)
python src/s04_stochastic.py ws          # wait-and-see, 20 single-scenario solves (~1 min)
for w in 0.5 0.75 1.0; do python src/s04_stochastic.py cvar $w; done   # CVaR frontier (~3 min each)
python - <<'PY'
import json
p = "results/tables/stoch_ckpt.json"; R = json.load(open(p))
R["designs"] = {"EV (= deterministic C)": R["y_EV"], "RP (w=0)": R["y_RPsolver"],
                "CVaR w=0.5": R["front"]["0.5"]["y"], "CVaR w=0.75": R["front"]["0.75"]["y"], "CVaR w=1": R["front"]["1.0"]["y"]}
json.dump(R, open(p, "w"), indent=1)
PY
for d in "EV (= deterministic C)" "RP (w=0)" "CVaR w=0.5" "CVaR w=0.75" "CVaR w=1"; do python src/s04_stochastic.py oos "$d"; done
python src/s04_stochastic.py base_oos
python src/s05_stochastic_figures.py
echo "Pipeline finished. Figures in results/figures, tables in results/tables."
