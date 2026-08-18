# space-3dgs

3D Gaussian Splatting for the interiors of crewed spacecraft, from imagery a
free-flying robot already collects. Given several passes of camera frames with
only *approximate* poses (robot localization, odometry), the pipeline

1. re-solves photometrically consistent camera poses with **prior-guided global
   structure-from-motion** (COLMAP features + GLOMAP), registered back into the
   robot's metric frame, and
2. optimizes an **anti-aliased Gaussian splat with one appearance code per capture
   sequence** and a culling schedule that keeps pruning to the end of training,

and hands you a `.ply` you can fly through in real time.

The included demo rebuilds the interior of the ISS Japanese Experiment Module
(JEM / Kibo) from five public [Astrobee](https://astrobee-iss-dataset.github.io/)
survey flights: 5,000 grayscale NavCam frames → **PSNR 31.2 / SSIM 0.918 /
LPIPS 0.208** on 625 held-out views at the native 1280×880, 1.2 M Gaussians.
Training on the re-solved poses instead of the robot's operational localization
poses is worth **+6.7 dB** on the identical evaluation set — the poses, not the
splat recipe, were the ceiling.

Everything is a plain [nerfstudio](https://docs.nerf.studio) method: `ns-train
space-3dgs`, `ns-viewer`, `ns-eval`, `ns-export` all work as usual.

---

## Contents

- [Method](#method)
- [Setup](#setup)
- [Quick start on your own capture](#quick-start-on-your-own-capture)
- [Demo: the ISS JEM module](#demo-the-iss-jem-module)
- [Repository layout](#repository-layout)
- [Configuration knobs](#configuration-knobs)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Method

### Stage 1 — prior-guided global SfM (`scripts/sfm_global.py`)

Onboard localization poses are metrically fine for navigation but
photometrically inconsistent at the level radiance-field training punishes: on
the ISS data the median disagreement with a self-consistent solve is 6.7 cm,
which at 0.9 m scene depth is ~45 px of contradictory supervision. So we re-solve
every camera and use the priors for only two things — proposing match
candidates and fixing the metric gauge:

| Step | What | Why |
|---|---|---|
| SIFT extraction | one shared camera from the dataset intrinsics | |
| Sequential matching | 10 neighbours along the name-sorted chain | cheap intra-pass overlap |
| **Position priors → spatial matching** | pairs proposed among images whose *prior* positions lie within 3 m (30 nearest) | supplies the cross-pass, cross-direction pairs a sequential chain never links; keeps candidates physically plausible in a corridor of look-alike racks |
| **GLOMAP global mapping** | rotation averaging → global positioning → a few rounds of BA | uses every verified edge at once, so degenerate low-parallax pairs (38 % of pairs on forward corridor motion) can't poison a seed the way they do in incremental SfM; and it is hours, not days, at 5 k images on CPU |
| Sim(3) registration | robust fit of SfM camera centres to the prior centres | puts poses *and* the sparse cloud back in the robot's metric frame |
| Export | `transforms.json` + `points3d.ply` (nerfstudio) + the COLMAP model | the cloud doubles as the splat initialization |

The priors never enter the solve itself; every match is still verified
geometrically. On the ISS data this registers all 5,000 frames in a single
model at 0.72 px mean reprojection error (1.5 M points).

### Stage 2 — the splat (`space_3dgs/`)

Built on splatfacto / gsplat with three deliberate changes:

- **Anti-aliased rasterization** — a corridor viewed along its axis shows the
  same rack faces at every scale; the mip filter keeps small/distant Gaussians
  from aliasing.
- **One achromatic appearance code per capture sequence**
  (`appearance_model.py`). Each sequence carries a learned 8-d code; a shared
  linear decoder maps it to a global gain and bias, squashed to ±0.25 log-gain
  and ±0.10 bias and applied identically to all channels — it can absorb
  per-flight exposure differences but *cannot invent colour casts* (a
  colour-capable variant did, on grayscale input). Codes start at zero and are
  tethered there by a ridge penalty, so the shared field carries the common
  appearance and codes carry only residuals. Held-out views use their own
  sequence's code; free viewer cameras use the mean code.
- **Pruning kept active after densification stops** (`cull_strategy.py`).
  Stock gsplat freezes *all* density control — including culling — when
  splitting stops (60 k of the 90 k steps here). That freezes late-fading
  haze into the export; keeping the opacity/scale prune running to the end
  gives a model whose median Gaussian opacity is 1.00 with < 1 % below 0.1.

Poses are held fixed (camera optimizer off): at sub-pixel pose consistency,
photometric pose refinement adds nothing and drifts the frame relative to the
fixed evaluation cameras. Everything else — L1 + 0.2·D-SSIM loss, learning
rates, split/clone thresholds (0.0008 gradient, 0.05 opacity cull) — is the
standard recipe. Sequence membership is read from the image filename
(`<sequence>_<digits>.<ext>`) by the `SequenceNerfstudio` dataparser; the
number of codes is inferred automatically.

---

## Setup

Two Python environments are the safe route: the tested nerfstudio 1.1.5 stack
runs on Python 3.8, while pycolmap ≥ 3.11 (which bundles GLOMAP) needs
Python ≥ 3.9. (A single 3.9/3.10 env may work; not tested.) Tested on Linux
with an RTX 3090 Ti (24 GB), CUDA 11.8, 16 CPU cores, 60 GB RAM.

**1. Training env** — nerfstudio 1.1.5 + gsplat 1.4.0 (follow the
[nerfstudio install guide](https://docs.nerf.studio/quickstart/installation.html)),
then register this method:

```bash
conda activate nerfstudio
git clone <this repo> space-3dgs && cd space-3dgs
pip install -e .            # installs the `space-3dgs` nerfstudio method (entry point)
ns-train space-3dgs --help  # should list --pipeline.model.num-sequences etc.
```

**2. SfM env** — any Python ≥ 3.9 with pycolmap:

```bash
conda create -n colmap python=3.12 -y && conda activate colmap
pip install "pycolmap>=3.11" numpy
python -c "import pycolmap; print(pycolmap.__version__)"   # 3.11+ (tested 4.1.1)
```

For the ISS demo, the download/prep scripts additionally need `pip install
gdown numpy` (and `opencv-python-headless` only if you want the optional seed
cloud from the prior poses).

The pycolmap wheel from PyPI is CPU-only for matching and mapping; that is what
the timings below assume. Extraction/matching can run on GPU with a
CUDA-enabled pycolmap build (`--device cuda`).

---

## Quick start on your own capture

**Input format.** A nerfstudio-style dataset with approximate poses:

```
data/my_scene_prior/
  transforms.json     top-level fl_x fl_y cx cy w h (k1 k2 p1 p2 optional),
                      frames[*].file_path = "images/<sequence>_<id>.<ext>",
                      frames[*].transform_matrix = 4x4 camera-to-world (OpenGL)
  images/             the frames (undistorted or with the OPENCV distortion above)
```

- All images must share one set of intrinsics.
- Name images `<sequence>_<digits>.<ext>` so that (a) sorting by name gives
  capture order within a sequence (sequential matching) and (b) each pass gets
  its own appearance code. Names without that pattern all fall into a single
  default sequence — that's fine for a plain single-pass capture.
- Poses need only be approximately right (their centres are used for pair
  proposals within `--spatial-max-distance`, and for the metric registration).
  If they are in metres, keep the defaults; otherwise scale
  `--spatial-max-distance` / `--align-max-error` accordingly.

**Solve poses** (SfM env; multi-hour for thousands of images on CPU, resumable):

```bash
conda activate colmap
python scripts/sfm_global.py --data data/my_scene_prior --output data/my_scene
#   -> data/my_scene/{transforms.json, images/, points3d.ply, colmap/sparse/0, sfm_report.json}
```

**Train / evaluate / export** (training env; ~50 min on a 3090 Ti for ~5 k images):

```bash
conda activate nerfstudio
bash scripts/train.sh data/my_scene my_scene
#   -> outputs/my_scene/space-3dgs/<timestamp>/   checkpoints + config.yml
#      outputs/my_scene/eval_metrics.json         PSNR / SSIM / LPIPS on every 8th frame
#      outputs/my_scene/export/splat.ply          the Gaussian splat
ns-viewer --load-config outputs/my_scene/space-3dgs/*/config.yml
```

`scripts/train.sh` is just `ns-train space-3dgs --data … && ns-eval && ns-export`;
any extra arguments go straight to `ns-train`, e.g. `--max-num-iterations 30000`
or `--vis viewer` to watch training live.

---

## Demo: the ISS JEM module

`examples/iss_jem/run.sh` reproduces the paper's reconstruction end to end from
the public Astrobee dataset. From the repo root, with both envs installed:

```bash
conda activate nerfstudio                       # ns-train on PATH
COLMAP_PY=~/anaconda3/envs/colmap/bin/python \  # python that has pycolmap
  bash examples/iss_jem/run.sh
```

What it does, step by step (each is idempotent — re-run to resume):

| Step | Command | Output | Budget |
|---|---|---|---|
| 1 | `download.py seq ff_return_journey_{forward,up,down,left,right}` | `data/raw/<flight>/` (5 same-day April-2021 survey flights through the JEM corridor, camera facing forward / up / down / left / right) | ~20 GB on disk after extraction |
| 2 | `prepare_sequence.py` | extracts `gray/<ts>.png` (undistorted 1280×880 NavCam frames), drops the archives | |
| 3 | `convert_to_nerfstudio.py --no-ply` | `data/iss_jem_prior/` — 5,000 frames after motion-based de-duplication (keep a frame only if the robot moved > 1.5 cm or turned > 0.75°), poses = the dataset's localization poses converted to OpenGL c2w | seconds |
| 4 | `sfm_global.py --data data/iss_jem_prior --output data/iss_jem` | re-solved poses + 1.5 M-point cloud, metric ISS frame; `sfm_report.json` records per-stage times, reprojection error, and how far the priors were off | ~5 h CPU on 16 cores (extract 8 min, sequential match 18 min, spatial match 27 min, GLOMAP 4.2 h), ~10 GB scratch (`SFM_ARGS="--work /fast/disk"` to move it) |
| 5 | `train.sh data/iss_jem iss_jem` | 90 k steps, eval on every 8th frame (625 views), `.ply` export | ~50 min GPU |

Expected result (paper configuration): PSNR ≈ 31.2, SSIM ≈ 0.918, LPIPS ≈ 0.208
at 1280×880; ~1.2 M Gaussians; SfM registers all 5,000 frames at ~0.72 px mean
reprojection error. Then:

```bash
ns-viewer --load-config outputs/iss_jem/space-3dgs/*/config.yml   # fly through the module
```

The Astrobee release also has other JEM sequences (`iva_kibo_trans`,
`iva_hatch_inspection*`, …; `python examples/iss_jem/download.py seq` lists
them). The demo deliberately uses only the five same-day flights: cargo moves
between visits, and mixing epochs ghosts the geometry.

---

## Repository layout

```
space_3dgs/                  the nerfstudio method (pip install -e . registers it)
  method_config.py             `space-3dgs` recipe: 90k steps, antialiased, poses fixed,
                               densify to 60k, cull to 90k, eval every 8th frame
  appearance_model.py          splatfacto + per-sequence achromatic gain/bias codes
  cull_strategy.py             gsplat DefaultStrategy that keeps pruning after refinement
  sequence_dataparser.py       nerfstudio dataparser + sequence_id from the filename
scripts/
  sfm_global.py                prior-guided global SfM  ->  nerfstudio dataset (pycolmap/GLOMAP)
  colmap_to_transforms.py      COLMAP model -> transforms.json + points3d.ply (+ optional Sim3)
  train.sh                     ns-train -> ns-eval -> ns-export for one run
  tcnn_shim/                   optional import shim, see Troubleshooting
examples/iss_jem/              the ISS demo: Astrobee download / frame prep /
                               localization-pose dataset builder, and run.sh
data/, outputs/                created by the scripts; git-ignored
```

---

## Configuration knobs

SfM (`scripts/sfm_global.py --help`):

| Flag | Default | Notes |
|---|---|---|
| `--spatial-max-distance`, `--spatial-neighbors` | 3.0, 30 | prior-distance window for cross-pass pair proposals; the JEM is ~8.5 m long |
| `--seq-overlap` | 10 | sequential-chain neighbours |
| `--min-num-matches` | 30 | GLOMAP view-graph edge threshold |
| `--keep-max-tracks` | 0 (off) | cap tracks in global positioning if GLOMAP runs out of RAM (1.5 M fits in 60 GB) |
| `--align-max-error` | 0.10 | RANSAC inlier radius (input units) for the metric registration |
| `--device` | auto | `cuda` for GPU SIFT/matching if your pycolmap build supports it |

Training (all standard `ns-train` overrides; the interesting ones):

| Flag | Default | Notes |
|---|---|---|
| `--pipeline.model.num-sequences` | 0 (= from the dataparser) | size of the appearance-code table |
| `--pipeline.model.appearance-embedding-dim` / `max-log-gain` / `max-bias` / `appearance-reg-mult` | 8 / 0.25 / 0.10 / 1e-4 | the appearance model |
| `--pipeline.model.cull-stop-iter` | 90000 | keep pruning until this step (0 = stock behaviour) |
| `--pipeline.model.stop-split-at` | 60000 | densification stops |
| `--pipeline.model.cull-alpha-thresh` / `densify-grad-thresh` | 0.05 / 0.0008 | near-stock; the permissive 0.01 / 0.0005 pair produced 30 % sub-0.1-opacity haze |
| `--pipeline.model.camera-optimizer.mode` | off | `SO3xR3` if your poses are *not* from the SfM stage |
| `--max-num-iterations` | 90000 | scale schedules with it if you shorten it a lot |
| `nerfstudio-data --eval-mode … --eval-interval …` | interval, 8 | the held-out split |

---

## Troubleshooting

- **`gsplat: No CUDA toolkit found`** then `'NoneType' object has no attribute
  'CameraModelType'`: gsplat JIT-compiles against the CUDA toolkit; if it lives
  inside the conda env, `export CUDA_HOME=$CONDA_PREFIX` and put
  `$CUDA_HOME/bin` on `PATH`.
- **nvcc rejects the host compiler** (`unsupported GNU version`): CUDA 11.8's
  nvcc needs GCC ≤ 11. Point it at one:
  `export CC=gcc-11 CXX=g++-11 NVCC_PREPEND_FLAGS="-ccbin g++-11"`.
- **`EnvironmentError` from tinycudann at import**: some tinycudann builds
  ship no extension for your GPU and raise an error nerfstudio doesn't catch.
  space-3dgs is gsplat-based and never needs tinycudann; put
  `scripts/tcnn_shim` on `PYTHONPATH` to shadow it out (`export
  PYTHONPATH=$PWD/scripts/tcnn_shim:$PYTHONPATH`).
- **GLOMAP killed (out of memory)** in global positioning: pass
  `--keep-max-tracks 1500000` (or lower). Fragmented view graphs — e.g. a
  sequential-only match set — produce millions of short tracks; on the ISS
  data the fix at full scale is exactly the spatial-matching stage, so check
  `sfm_report.json` shows `match_spatial_s` ran.
- **SfM registers only part of the images / several models**: increase
  `--spatial-max-distance` or `--spatial-neighbors` (more cross-pass edges),
  or check that the prior positions really are in the units you think.
- **RAM during training**: the datamanager caches all training images as
  uint8; 5 k frames at 1280×880 is ~17 GB. Subsample or downscale
  (`nerfstudio-data --downscale-factor 2`) on smaller machines.
- **`ns-eval` PSNR looks off vs the viewer**: evaluation cameras use their own
  sequence's appearance code, the viewer uses the mean code — that's expected.

---

Built on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio),
[gsplat](https://github.com/nerfstudio-project/gsplat),
[COLMAP](https://colmap.github.io/) / [pycolmap](https://github.com/colmap/colmap)
and [GLOMAP](https://github.com/colmap/glomap). ISS imagery: the
[Astrobee ISS Free-Flyer Datasets](https://astrobee-iss-dataset.github.io/)
(NASA Ames).

License: TBD.
