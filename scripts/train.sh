#!/usr/bin/env bash
# Train, evaluate, and export one space-3dgs run.
#
# Usage: bash scripts/train.sh <dataset-dir> <experiment-name> [extra ns-train args...]
#   e.g. bash scripts/train.sh data/iss_jem iss_jem
#        bash scripts/train.sh data/my_scene my_scene --max-num-iterations 30000
#
# Writes outputs/<experiment-name>/space-3dgs/<timestamp>/  (checkpoints, config.yml)
#        outputs/<experiment-name>/eval_metrics.json          (held-out PSNR/SSIM/LPIPS)
#        outputs/<experiment-name>/export/splat.ply           (the Gaussian splat)
# Requires the nerfstudio env with `pip install -e .` done (ns-train on PATH).
set -euo pipefail
data=${1:?usage: train.sh <dataset-dir> <experiment-name> [ns-train args...]}
exp=${2:?usage: train.sh <dataset-dir> <experiment-name> [ns-train args...]}
shift 2

echo "=== [$exp] train ==="
ns-train space-3dgs \
  --data "$data" \
  --output-dir outputs \
  --experiment-name "$exp" \
  --viewer.quit-on-train-completion True \
  "$@"

CFG=$(ls -t "outputs/$exp"/space-3dgs/*/config.yml | head -1)
echo "=== [$exp] eval ($CFG) ==="
ns-eval --load-config "$CFG" --output-path "outputs/$exp/eval_metrics.json"

echo "=== [$exp] export ==="
ns-export gaussian-splat --load-config "$CFG" --output-dir "outputs/$exp/export"

echo "=== [$exp] summary ==="
python -m json.tool "outputs/$exp/eval_metrics.json" | grep -E '"(psnr|ssim|lpips)"' || true
echo "view:   space-3dgs-viewer outputs/$exp    (or: ns-viewer --load-config $CFG)"
