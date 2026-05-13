#!/bin/bash
set -euo pipefail

echo "============================================================"
echo " CE-AIS MVP: Full Pipeline"
echo " $(date)"
echo "============================================================"

cd "$(dirname "$0")/../.."

echo ""
echo ">>> Step 0: Environment Check"
uv run python scripts/mvp/00_check_env.py || exit 1

echo ""
echo ">>> Step 1: Train BC Proxy (~5-10 min)"
uv run python scripts/mvp/01_train_bc.py || exit 1

echo ""
echo ">>> Step 2: Train Encoder + CE-WM (~30-60 min)"
uv run python scripts/mvp/02_train_cewm.py || exit 1

echo ""
echo ">>> Step 3: Evaluate (~5-10 min)"
uv run python scripts/mvp/03_eval_mvp.py || exit 1

echo ""
echo "============================================================"
echo " MVP Pipeline Complete!"
echo " Results: results/mvp/mvp_results.json"
echo " $(date)"
echo "============================================================"
