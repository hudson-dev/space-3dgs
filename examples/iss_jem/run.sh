#!/usr/bin/env bash
# End-to-end ISS demo: five Astrobee survey flights through the JEM module
# -> nerfstudio dataset with the robot's localization poses (priors)
# -> prior-guided global SfM (re-solved poses + seed cloud)
# -> space-3dgs training, held-out evaluation, .ply export.
#
# Usage:  bash examples/iss_jem/run.sh            (from the repo root)
# Env:    COLMAP_PY  python with pycolmap>=3.11 (default: python on PATH)
#         SFM_ARGS   extra args for scripts/sfm_global.py (e.g. "--work /fast/scratch")
#
# Budget (RTX 3090 Ti, 16-core CPU): ~20 GB of frames on disk, SfM ~5 h CPU,
# training ~50 min GPU. Every step is resumable; re-run to continue.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
COLMAP_PY=${COLMAP_PY:-python}
SEQS=(ff_return_journey_forward ff_return_journey_up ff_return_journey_down
      ff_return_journey_left ff_return_journey_right)

echo "=== 1/5 download (${SEQS[*]}) ==="
python examples/iss_jem/download.py seq "${SEQS[@]}"

echo "=== 2/5 extract frames ==="
python examples/iss_jem/prepare_sequence.py "${SEQS[@]/#/data/raw/}"

echo "=== 3/5 nerfstudio dataset with localization-pose priors ==="
if [ ! -f data/iss_jem_prior/transforms.json ]; then
  python examples/iss_jem/convert_to_nerfstudio.py \
    --out data/iss_jem_prior --seqs "${SEQS[@]/#/data/raw/}" --no-ply
fi

echo "=== 4/5 prior-guided global SfM ==="
"$COLMAP_PY" scripts/sfm_global.py --data data/iss_jem_prior --output data/iss_jem ${SFM_ARGS:-}

echo "=== 5/5 train / eval / export ==="
bash scripts/train.sh data/iss_jem iss_jem
