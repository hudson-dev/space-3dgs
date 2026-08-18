"""Live viser viewer for space-3dgs splats.

Renders a Gaussian splat straight in the browser with viser's WebGL splat
renderer -- no GPU, no nerfstudio pipeline, no dataset images -- from either

  * an exported splat   outputs/<exp>/export/splat.ply         (ns-export gaussian-splat,
                        or any 3DGS-style .ply)
  * a training run      outputs/<exp>/space-3dgs/<ts>/         (config.yml, a .ckpt, or the
                        run dir; also plain outputs/<exp> = its latest run): the Gaussians
                        are read straight from the newest checkpoint

and overlays the training cameras (one colour per capture sequence; click a
frustum to jump into that view) and the SfM seed cloud, moved into the splat's
frame with the run's dataparser_transforms.json.

--watch keeps polling the source and hot-swaps the splat when a new step-*.ckpt
lands (or the .ply is rewritten), so you can start it before / next to
`ns-train` and follow training as checkpoints are written (every
--steps-per-save steps, 15k by default -- lower it for a denser live view).

  space-3dgs-viewer outputs/iss_jem                       # latest run of that experiment
  space-3dgs-viewer outputs/iss_jem/export/splat.ply      # an exported splat
  space-3dgs-viewer outputs/iss_jem --watch               # follow a training run live
  python -m space_3dgs.viewer --help                      # same thing without the entry point

Colour is the SH DC term (view-independent) with the appearance correction of a
chosen capture sequence (or the mean code, as nerfstudio's viewer uses) folded
into it; opacities and scales are the trained values. For exact model renders
(full SH, antialiasing, per-view appearance) use `ns-viewer` / `ns-train --vis
viewer`, which render through the model on the GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SH_C0 = 0.28209479177387814
MIN_SCALE = 1e-4  # floor on Gaussian scale (scene units) so covariances stay positive definite
LEGACY_APPLIED_TRANSFORM = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]], dtype=np.float64)
MEAN_APPEARANCE = "mean (viewer default)"
_UNSET = object()
PALETTE = [  # one colour per capture sequence (matplotlib tab10)
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]


def log(msg: str) -> None:
    print(f"[viewer {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def short(path: Path) -> str:
    """Path relative to the working directory when it lives underneath it."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- PLY


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply_vertices(path: Path) -> Dict[str, np.ndarray]:
    """Return the vertex element of a PLY file as {property: array}.

    Minimal reader (no dependency): binary little/big endian or ascii, scalar
    properties. Enough for 3DGS splat exports and the SfM point cloud.
    """
    with open(path, "rb") as f:
        header: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            header.append(line.decode("ascii", "replace").strip())
            if header[-1] == "end_header":
                break
        if not header or header[0] != "ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = next(l.split()[1] for l in header if l.startswith("format "))
        elements: List[Tuple[str, int, List[Tuple[str, Optional[str]]]]] = []
        for l in header:
            parts = l.split()
            if not parts:
                continue
            if parts[0] == "element":
                elements.append((parts[1], int(parts[2]), []))
            elif parts[0] == "property":
                if parts[1] == "list":
                    elements[-1][2].append((parts[-1], None))
                else:
                    elements[-1][2].append((parts[-1], _PLY_TYPES[parts[1]]))
        endian = ">" if fmt == "binary_big_endian" else "<"
        for name, count, props in elements:
            if any(dt is None for _, dt in props):
                raise ValueError(f"{path}: list-typed properties in element '{name}' are not supported")
            if fmt == "ascii":
                rows = np.loadtxt([f.readline().decode("ascii", "replace") for _ in range(count)],
                                  dtype=np.float64, ndmin=2)
                data = {p: rows[:, i] for i, (p, _) in enumerate(props)}
            else:
                dtype = np.dtype([(p, endian + dt) for p, dt in props])
                arr = np.fromfile(f, dtype=dtype, count=count)
                if arr.shape[0] != count:
                    raise ValueError(f"{path}: truncated PLY data (element '{name}')")
                data = {p: arr[p] for p, _ in props}
            if name == "vertex":
                return data
    raise ValueError(f"{path}: no 'vertex' element")


# --------------------------------------------------------------------------- splat


@dataclass
class Appearance:
    """The per-sequence achromatic gain/bias model of SequenceAppearanceModel."""

    codes: np.ndarray  # (S, D)
    weight: np.ndarray  # (2, D)
    bias: np.ndarray  # (2,)
    max_log_gain: float = 0.25
    max_bias: float = 0.10

    @property
    def num_sequences(self) -> int:
        return int(self.codes.shape[0])

    def gain_bias(self, sequence: Optional[int]) -> Tuple[float, float]:
        """Exposure gain/bias for one sequence, or for the mean code (None)."""
        code = self.codes.mean(axis=0) if sequence is None else self.codes[sequence]
        raw = self.weight @ code + self.bias
        return float(np.exp(self.max_log_gain * np.tanh(raw[0]))), float(self.max_bias * np.tanh(raw[1]))


@dataclass
class Splat:
    means: np.ndarray  # (N, 3)
    covariances: np.ndarray  # (N, 3, 3)
    rgbs: np.ndarray  # (N, 3) in [0, 1], SH DC term, no appearance correction
    opacities: np.ndarray  # (N, 1)
    source: Path
    step: Optional[int] = None
    total: int = 0  # count before --min-opacity / --max-gaussians
    appearance: Optional[Appearance] = None

    def __len__(self) -> int:
        return int(self.means.shape[0])


def quat_scale_to_covariance(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """R(q) diag(s^2) R(q)^T for wxyz quaternions and linear scales."""
    q = quats / np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    M = R * scales[:, None, :]  # R @ diag(s)
    return (M @ M.transpose(0, 2, 1)).astype(np.float32)


def _finish_splat(
    means: np.ndarray, scales: np.ndarray, quats: np.ndarray, opacities: np.ndarray, rgbs: np.ndarray,
    source: Path, step: Optional[int], appearance: Optional[Appearance],
    min_opacity: float, max_gaussians: int,
) -> Splat:
    total = means.shape[0]
    keep = (
        np.isfinite(means).all(1) & np.isfinite(scales).all(1) & np.isfinite(quats).all(1)
        & np.isfinite(opacities) & (opacities >= min_opacity)
    )
    idx = np.flatnonzero(keep)
    if 0 < max_gaussians < idx.size:
        # keep the most opaque ones (deterministic; drops the faint haze first)
        idx = idx[np.argsort(-opacities[idx], kind="stable")[:max_gaussians]]
        idx.sort()
    # Mid-training checkpoints can hold collapsed Gaussians (scale -> 0); viser
    # Cholesky-factors the covariances, so keep them strictly positive definite.
    covariances = quat_scale_to_covariance(quats[idx], np.maximum(scales[idx], MIN_SCALE))
    try:
        np.linalg.cholesky(covariances.astype(np.float64) + 1e-7)
    except np.linalg.LinAlgError:
        good = np.linalg.eigvalsh(covariances.astype(np.float64)).min(axis=1) > 0
        log(f"dropping {int((~good).sum())} Gaussians with degenerate covariance")
        idx, covariances = idx[good], covariances[good]
    return Splat(
        means=means[idx].astype(np.float32),
        covariances=covariances,
        rgbs=np.clip(rgbs[idx], 0.0, 1.0).astype(np.float32),
        opacities=np.clip(opacities[idx], 0.0, 1.0).astype(np.float32)[:, None],
        source=source, step=step, total=total, appearance=appearance,
    )


def splat_from_ply(path: Path, min_opacity: float = 0.0, max_gaussians: int = 0) -> Splat:
    """A 3DGS-style .ply (ns-export gaussian-splat, or the original 3DGS layout)."""
    v = read_ply_vertices(path)
    need = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [k for k in need if k not in v]
    if missing:
        raise ValueError(f"{path}: not a Gaussian-splat PLY (missing {missing})")
    means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    if "f_dc_0" in v:
        rgbs = 0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], -1)
    elif "red" in v:  # ns-export --ply-color-mode rgb
        rgbs = np.stack([v["red"], v["green"], v["blue"]], -1).astype(np.float64)
        if v["red"].dtype.kind in "ui":
            rgbs /= 255.0
    else:
        raise ValueError(f"{path}: no colour (f_dc_* or red/green/blue)")
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float64))
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float64)
    opacities = sigmoid(v["opacity"].astype(np.float64))
    return _finish_splat(means, scales, quats, opacities, rgbs, path, None, None, min_opacity, max_gaussians)


def splat_from_checkpoint(
    path: Path, run_cfg: Optional[dict] = None, min_opacity: float = 0.0, max_gaussians: int = 0
) -> Splat:
    """A nerfstudio checkpoint: reads gauss_params (and the appearance model) directly."""
    import torch  # only needed for checkpoints; .ply viewing works without it

    ckpt = torch.load(str(path), map_location="cpu")
    sd = ckpt["pipeline"] if "pipeline" in ckpt else ckpt
    key = next((k for k in sd if k.endswith("gauss_params.means")), None)
    if key is not None:
        prefix = key[: -len("means")]
    else:  # pre-1.1 splatfacto checkpoints stored the parameters flat
        key = next((k for k in sd if k.endswith(".means") and k[: -len("means")] + "quats" in sd), None)
        if key is None:
            raise ValueError(f"{path}: no Gaussian parameters (gauss_params.means) in the checkpoint")
        prefix = key[: -len("means")]
    model_prefix = prefix[: -len("gauss_params.")] if prefix.endswith("gauss_params.") else prefix

    def get(name: str) -> np.ndarray:
        return sd[prefix + name].detach().float().cpu().numpy().astype(np.float64)

    means = get("means")
    scales = np.exp(get("scales"))
    quats = get("quats")
    opacities = sigmoid(get("opacities")).reshape(-1)
    dc = get("features_dc")
    sh_degree = int((run_cfg or {}).get("sh_degree", 3))
    rgbs = 0.5 + SH_C0 * dc if sh_degree > 0 else sigmoid(dc)

    appearance = None
    if model_prefix + "appearance_embedding.weight" in sd:
        appearance = Appearance(
            codes=sd[model_prefix + "appearance_embedding.weight"].detach().float().cpu().numpy(),
            weight=sd[model_prefix + "appearance_decoder.weight"].detach().float().cpu().numpy(),
            bias=sd[model_prefix + "appearance_decoder.bias"].detach().float().cpu().numpy(),
            max_log_gain=float((run_cfg or {}).get("max_log_gain", 0.25)),
            max_bias=float((run_cfg or {}).get("max_bias", 0.10)),
        )
    step = ckpt.get("step") if isinstance(ckpt, dict) else None
    return _finish_splat(means, scales, quats, opacities, rgbs, path, step, appearance, min_opacity, max_gaussians)


# --------------------------------------------------------------------------- run / dataset


def sequence_name(image_path: Path) -> str:
    """``<sequence>_<digits>.<ext>`` -> ``<sequence>`` (same rule as the dataparser)."""
    stem = image_path.stem
    prefix, separator, suffix = stem.rpartition("_")
    if not separator or not suffix.isdigit() or not prefix:
        return "default"
    return prefix


def read_run_config(run_dir: Path) -> dict:
    """The few fields of a run's config.yml we care about, without importing nerfstudio.

    Returns {} if there is no config.yml. Keys (when found): data, sh_degree,
    max_log_gain, max_bias, method.
    """
    cfg_path = run_dir / "config.yml"
    out: dict = {}
    if not cfg_path.exists():
        return out
    text = cfg_path.read_text()
    m = re.search(r"^\s*data: (?:&\w+ )?!!python/object/apply:pathlib\.\w+Path\n((?:\s*- .*\n)+)", text, re.M)
    if m:
        parts = [line.strip()[2:].strip().strip("'\"") for line in m.group(1).splitlines() if line.strip()]
        out["data"] = Path(*parts) if parts else None
    for key in ("sh_degree", "max_log_gain", "max_bias", "eval_interval"):
        m = re.search(rf"^\s*{key}: ([-+0-9.eE]+)\s*$", text, re.M)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"^method_name: (\S+)", text, re.M)
    if m:
        out["method"] = m.group(1)
    if "data" not in out:  # unusual layout: let yaml rebuild the config objects if it can
        try:
            import yaml

            cfg = yaml.load(text, Loader=yaml.Loader)  # noqa: S506 - our own training config
            data = getattr(cfg, "data", None) or getattr(cfg.pipeline.datamanager.dataparser, "data", None)
            if data is not None:
                out["data"] = Path(data)
            model = cfg.pipeline.model
            for key in ("sh_degree", "max_log_gain", "max_bias"):
                if hasattr(model, key):
                    out.setdefault(key, float(getattr(model, key)))
        except Exception:  # noqa: BLE001 - best effort, the fields all have defaults
            pass
    return out


@dataclass
class DataparserFrame:
    """Maps dataset (transforms.json) coordinates into the frame the splat was trained in.

    nerfstudio's dataparser_transforms.json stores the transform from the
    *original* data frame; when the dataset carries an ``applied_transform``
    (or nerfstudio assumes the legacy one because ``<data>/colmap/sparse/0``
    exists) that has to be undone first, exactly as the dataparser does.
    """

    transform: np.ndarray  # (4, 4) saved-coordinates -> splat frame (before scaling)
    scale: float  # dataparser scale (as saved)
    pose_scale: float  # scale applied to camera centres

    @classmethod
    def load(cls, run_dir: Path, meta: Optional[dict], data_arg: Optional[Path]) -> Optional["DataparserFrame"]:
        path = run_dir / "dataparser_transforms.json"
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        T = np.eye(4)
        T[:3, :4] = np.asarray(saved["transform"], dtype=np.float64)
        scale = float(saved.get("scale", 1.0))
        applied = None
        if meta is not None and "applied_transform" in meta:
            applied = np.asarray(meta["applied_transform"], dtype=np.float64)
        elif data_arg is not None and (data_arg / "colmap/sparse/0").exists():
            applied = LEGACY_APPLIED_TRANSFORM
        if applied is not None:
            A = np.eye(4)
            A[:3, :4] = applied
            T = T @ np.linalg.inv(A)
        applied_scale = float(meta.get("applied_scale", 1.0)) if meta else 1.0
        return cls(transform=T, scale=scale, pose_scale=scale / applied_scale)

    def poses(self, c2w: np.ndarray) -> np.ndarray:  # (M, 4, 4) -> (M, 4, 4)
        out = self.transform[None] @ c2w
        out[:, :3, 3] *= self.pose_scale
        return out

    def points(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz @ self.transform[:3, :3].T + self.transform[:3, 3]) * self.scale


@dataclass
class Dataset:
    c2w: np.ndarray  # (M, 4, 4), OpenGL camera-to-world, in the splat frame
    fov_y: np.ndarray  # (M,)
    aspect: np.ndarray  # (M,)
    names: List[str]
    sequence: np.ndarray  # (M,) int
    sequence_names: List[str]
    ply_path: Optional[Path]
    frame: Optional[DataparserFrame]
    total: int

    _points: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)

    def points(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(xyz, rgb uint8) of the SfM seed cloud in the splat frame, loaded on first use."""
        if self._points is None and self.ply_path is not None and self.ply_path.exists():
            v = read_ply_vertices(self.ply_path)
            xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
            if "red" in v:
                rgb = np.stack([v["red"], v["green"], v["blue"]], -1)
                rgb = rgb.astype(np.uint8) if rgb.dtype.kind in "ui" else (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            else:
                rgb = np.full((xyz.shape[0], 3), 200, dtype=np.uint8)
            if self.frame is not None:
                xyz = self.frame.points(xyz)
            self._points = (xyz.astype(np.float32), rgb)
        return self._points


def load_dataset(data: Path, run_dir: Optional[Path], max_cameras: int) -> Dataset:
    transforms_path = data if data.suffix == ".json" else data / "transforms.json"
    meta = json.loads(transforms_path.read_text())
    data_dir = transforms_path.parent
    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])
    names = [fr["file_path"] for fr in frames]
    seq_names = sorted({sequence_name(Path(n)) for n in names})
    seq_index = {s: i for i, s in enumerate(seq_names)}

    def intr(fr: dict, key: str) -> float:
        return float(fr.get(key, meta.get(key)))

    c2w = np.array([fr["transform_matrix"] for fr in frames], dtype=np.float64)
    if c2w.shape[1] == 3:
        c2w = np.concatenate([c2w, np.tile(np.array([[[0, 0, 0, 1.0]]]), (len(frames), 1, 1))], axis=1)
    fov_y = np.array([2 * np.arctan2(intr(fr, "h"), 2 * intr(fr, "fl_y")) for fr in frames])
    aspect = np.array([intr(fr, "w") / intr(fr, "h") for fr in frames])
    seq = np.array([seq_index[sequence_name(Path(n))] for n in names], dtype=np.int64)

    frame = None
    if run_dir is not None:
        # nerfstudio decides on the legacy applied_transform from the --data path it was trained with
        frame = DataparserFrame.load(run_dir, meta, data)
        if frame is not None:
            c2w = frame.poses(c2w)
    total = len(frames)
    if 0 < max_cameras < total:
        stride = int(np.ceil(total / max_cameras))
        keep = np.arange(0, total, stride)
        c2w, fov_y, aspect, seq = c2w[keep], fov_y[keep], aspect[keep], seq[keep]
        names = [names[i] for i in keep]
    ply_rel = meta.get("ply_file_path")
    ply_path = (data_dir / ply_rel) if ply_rel else (data_dir / "points3d.ply")
    return Dataset(c2w, fov_y, aspect, names, seq, seq_names, ply_path if ply_path.exists() else None, frame, total)


# --------------------------------------------------------------------------- source resolution


def _step_of(ckpt: Path) -> int:
    m = re.search(r"step-(\d+)", ckpt.name)
    return int(m.group(1)) if m else -1


class Source:
    """What to show and where the newest version of it lives."""

    def __init__(self, path: Path, run_override: Optional[Path] = None):
        self.path = path.resolve()
        # kind: "ply" | "ckpt" | "run" | "search"
        self.run_dir: Optional[Path] = None
        if self.path.is_file() and self.path.suffix == ".ply":
            self.kind = "ply"
            self.run_dir = self._guess_run(self.path)
        elif self.path.is_file() and self.path.suffix == ".ckpt":
            self.kind = "ckpt"
            self.run_dir = self.path.parent.parent
        elif self.path.is_file() and self.path.name == "config.yml":
            self.kind = "run"
            self.run_dir = self.path.parent
        elif self.path.is_dir() and ((self.path / "nerfstudio_models").exists() or (self.path / "config.yml").exists()):
            self.kind = "run"
            self.run_dir = self.path
        elif self.path.is_dir() or not self.path.exists():
            self.kind = "search"
            self.run_dir = self._latest_run(self.path)
        else:
            raise SystemExit(f"don't know how to view {self.path} (expect .ply / .ckpt / config.yml / a run or experiment dir)")
        if run_override is not None:
            self.run_dir = run_override.resolve()

    @staticmethod
    def _latest_run(root: Path) -> Optional[Path]:
        cfgs = [p for depth in ("*/config.yml", "*/*/config.yml", "*/*/*/config.yml") for p in root.glob(depth)]
        return max(cfgs, key=lambda p: p.stat().st_mtime).parent if cfgs else None

    @classmethod
    def _guess_run(cls, ply: Path) -> Optional[Path]:
        # outputs/<exp>/export/splat.ply  ->  outputs/<exp>/<method>/<timestamp>/
        for root in (ply.parent, ply.parent.parent):
            if (root / "config.yml").exists():
                return root
            run = cls._latest_run(root)
            if run is not None:
                return run
        return None

    def latest(self) -> Optional[Path]:
        """Newest file to load, or None if nothing exists yet."""
        if self.kind in ("ply", "ckpt"):
            return self.path if self.path.exists() else None
        if self.kind == "search":
            self.run_dir = self._latest_run(self.path)
        if self.run_dir is None:
            return None
        ckpts = [p for p in (self.run_dir / "nerfstudio_models").glob("step-*.ckpt")]
        return max(ckpts, key=_step_of) if ckpts else None

    def describe(self) -> str:
        kind = {"ply": "splat", "ckpt": "checkpoint", "run": "run", "search": "latest run under"}[self.kind]
        return f"{kind} {short(self.path)}"


@dataclass(frozen=True)
class Version:
    path: Path
    mtime_ns: int
    size: int

    @classmethod
    def of(cls, path: Path) -> "Version":
        st = path.stat()
        return cls(path, st.st_mtime_ns, st.st_size)


# --------------------------------------------------------------------------- viewer


class Viewer:
    def __init__(self, args: argparse.Namespace):
        import viser

        self.args = args
        self.source = Source(Path(args.path), Path(args.run) if args.run else None)
        self.server = viser.ViserServer(host=args.host, port=args.port, label="space-3dgs")
        self.server.scene.set_up_direction("+z")
        self._add_splats = getattr(self.server.scene, "add_gaussian_splats", None) or getattr(
            self.server.scene, "_add_gaussian_splats"
        )
        self.lock = threading.Lock()  # splat swap
        self.check_lock = threading.Lock()  # one check()/reload at a time (poll loop vs. Reload button)
        self.cam_lock = threading.RLock()  # frustum / point-cloud redraws
        self.splat: Optional[Splat] = None
        self.version: Optional[Version] = None
        self.splat_handle = None
        self.dataset: Optional[Dataset] = None
        self.dataset_run: object = _UNSET  # run dir the dataset/run_cfg were read for
        self.frustums: List = []
        self.points_handle = None
        self.home: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.run_cfg: dict = {}
        self._build_gui()
        self.server.on_client_connect(self._on_connect)

    # ---- GUI

    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.configure_theme(control_layout="collapsible", show_logo=False, brand_color=(80, 140, 200))
        self.status = gui.add_markdown("_waiting for a splat …_")
        with gui.add_folder("Splat"):
            self.cb_splat = gui.add_checkbox("Show splat", True)
            self.dd_appearance = gui.add_dropdown(
                "Appearance", [MEAN_APPEARANCE], MEAN_APPEARANCE, visible=False,
                hint="Fold one sequence's learned exposure gain/bias into the colours",
            )
            self.cb_watch = gui.add_checkbox(
                "Watch for updates", bool(self.args.watch),
                hint=f"Poll the source every {self.args.poll_interval:g}s and reload new checkpoints",
            )
            self.btn_reload = gui.add_button("Reload now", icon=None)
        with gui.add_folder("Cameras & points", expand_by_default=True):
            self.cb_cameras = gui.add_checkbox("Show training cameras", not self.args.no_cameras)
            self.dd_frustum = gui.add_dropdown("Frustum size", ["0.5×", "1×", "2×", "4×"], "1×")
            self.cb_points = gui.add_checkbox("Show SfM seed cloud", False, hint="points3d.ply of the dataset")
            self.sl_point_size = gui.add_slider("Point size", 0.001, 0.05, 0.001, 0.004)
            self.seq_folder = gui.add_folder("Sequences", expand_by_default=False)
            self.seq_checkboxes: Dict[int, object] = {}
        with gui.add_folder("View"):
            self.btn_home = gui.add_button("Reset view")

        self.cb_splat.on_update(lambda _: self._set_splat_visible(self.cb_splat.value))
        self.dd_appearance.on_update(lambda _: self._push_splat())
        self.btn_reload.on_click(lambda _: self.check(force=True))
        self.cb_cameras.on_update(lambda _: self._set_cameras_visible())
        self.dd_frustum.on_update(lambda _: self._draw_cameras())
        self.cb_points.on_update(lambda _: self._draw_points())
        self.sl_point_size.on_update(lambda _: self._draw_points())
        self.btn_home.on_click(lambda ev: self._go_home(ev.client))

    def _set_status(self, extra: str = "") -> None:
        s = self.splat
        lines = [f"**Source** `{self.source.describe()}`"]
        if s is None:
            lines.append("_no splat loaded yet_" + (" — waiting …" if self.cb_watch.value else ""))
        else:
            what = f"**step {s.step}**" if s.step is not None else f"`{s.source.name}`"
            kept = f"{len(s):,}" + (f" of {s.total:,}" if len(s) != s.total else "")
            lines.append(f"{what} · {kept} Gaussians · loaded {time.strftime('%H:%M:%S')}")
        if self.dataset is not None:
            d = self.dataset
            shown = f"{len(d.names):,}" + (f" of {d.total:,}" if len(d.names) != d.total else "")
            lines.append(f"{shown} cameras · {len(d.sequence_names)} sequence(s)"
                         + ("" if d.frame is not None else " · **frame unknown** (no dataparser_transforms.json)"))
        if extra:
            lines.append(extra)
        self.status.content = "  \n".join(lines)

    # ---- splat

    def _current_rgbs(self) -> np.ndarray:
        assert self.splat is not None
        s = self.splat
        if s.appearance is None or self.dd_appearance.value == MEAN_APPEARANCE:
            gain, bias = s.appearance.gain_bias(None) if s.appearance is not None else (1.0, 0.0)
        else:
            gain, bias = s.appearance.gain_bias(self.dd_appearance.options.index(self.dd_appearance.value) - 1)
        if gain == 1.0 and bias == 0.0:
            return s.rgbs
        return np.clip(s.rgbs * gain + bias, 0.0, 1.0).astype(np.float32)

    def _push_splat(self) -> None:
        """(Re)upload the current splat to every client."""
        with self.lock:
            if self.splat is None:
                return
            if self.splat_handle is not None:
                self.splat_handle.remove()
            self.splat_handle = self._add_splats(
                "/splat",
                centers=self.splat.means,
                covariances=self.splat.covariances,
                rgbs=self._current_rgbs(),
                opacities=self.splat.opacities,
                visible=self.cb_splat.value,
            )

    def _set_splat_visible(self, visible: bool) -> None:
        if self.splat_handle is not None:
            self.splat_handle.visible = visible

    def _load(self, path: Path) -> Splat:
        a = self.args
        if path.suffix == ".ply":
            return splat_from_ply(path, a.min_opacity, a.max_gaussians)
        return splat_from_checkpoint(path, self.run_cfg, a.min_opacity, a.max_gaussians)

    def check(self, force: bool = False) -> bool:
        """Load the newest version of the source if it changed. Returns True if reloaded."""
        if not self.check_lock.acquire(blocking=False):
            return False  # a reload is already in progress
        try:
            return self._check(force)
        finally:
            self.check_lock.release()

    def _check(self, force: bool) -> bool:
        path = self.source.latest()
        if self.dataset_run is _UNSET or self.dataset_run != self.source.run_dir:  # first call / newest run changed
            self.run_cfg = read_run_config(self.source.run_dir) if self.source.run_dir is not None else {}
            self._ensure_dataset()
        if path is None:
            self._set_status()
            return False
        try:
            version = Version.of(path)
        except FileNotFoundError:  # nerfstudio deleted the previous checkpoint under us
            return False
        if not force and version == self.version:
            return False
        age = time.time() - version.mtime_ns / 1e9
        if age < self.args.settle:  # probably still being written
            if not force:
                return False
            time.sleep(self.args.settle - age)
        t0 = time.time()
        try:
            splat = self._load(path)
        except Exception as e:  # noqa: BLE001 - half-written file, wrong format, …: retried next poll / Reload
            log(f"could not load {short(path)}: {type(e).__name__}: {e}")
            self._set_status(f"⚠ could not load `{path.name}`: {e}")
            return False
        self.version = version
        with self.lock:
            self.splat = splat
        options = [MEAN_APPEARANCE]
        if splat.appearance is not None:
            names = self.dataset.sequence_names if self.dataset is not None else []
            if len(names) != splat.appearance.num_sequences:
                names = [f"sequence {i}" for i in range(splat.appearance.num_sequences)]
            options += names
        if list(self.dd_appearance.options) != options:
            self.dd_appearance.options = options
            self.dd_appearance.value = MEAN_APPEARANCE
        self.dd_appearance.visible = splat.appearance is not None
        self._push_splat()
        self._update_home()
        self._set_status()
        step = f" step {splat.step}" if splat.step is not None else ""
        log(f"loaded {short(path)}{step}: {len(splat):,} Gaussians in {time.time() - t0:.1f}s")
        if len(splat) > 1_500_000 and not self.args.max_gaussians:
            log("that is a lot for the browser; --max-gaussians 1000000 (or --min-opacity) lightens it")
        return True

    # ---- cameras / points

    def _ensure_dataset(self) -> None:
        run_dir = self.source.run_dir
        self.dataset_run = run_dir
        data = Path(self.args.data) if self.args.data else self.run_cfg.get("data")
        if data is None:
            if run_dir is not None:
                log("no dataset known (config.yml has no data path); pass --data for the camera overlay")
            return
        data = Path(data)
        if not data.exists() and run_dir is not None:  # relative --data recorded from another cwd
            for base in [run_dir] + list(run_dir.parents)[:4]:
                if (base / data).exists():
                    data = base / data
                    break
        if not data.exists():
            log(f"dataset {data} not found; no camera overlay (pass --data)")
            return
        try:
            self.dataset = load_dataset(data, run_dir, self.args.max_cameras)
        except Exception as e:  # noqa: BLE001
            log(f"could not read dataset {data}: {type(e).__name__}: {e}")
            return
        d = self.dataset
        if d.frame is None and run_dir is not None:
            log(f"no dataparser_transforms.json in {run_dir}: cameras drawn in the dataset frame, "
                "which only matches the splat if the dataparser applied no transform")
        log(f"dataset {data}: {d.total:,} cameras ({len(d.names):,} drawn), sequences {d.sequence_names}")
        # per-sequence toggles
        for cb in self.seq_checkboxes.values():
            cb.remove()
        self.seq_checkboxes = {}
        with self.seq_folder:
            for i, name in enumerate(d.sequence_names):
                cb = self.server.gui.add_checkbox(name, True, hint=f"colour {PALETTE[i % len(PALETTE)]}")
                cb.on_update(lambda _: self._set_cameras_visible())
                self.seq_checkboxes[i] = cb
        self._draw_cameras()
        self._draw_points()

    def _frustum_scale(self) -> float:
        d = self.dataset
        base = self.args.frustum_scale
        if base <= 0 and d is not None:
            centres = d.c2w[:, :3, 3]
            base = 0.02 * float(np.linalg.norm(centres.max(0) - centres.min(0))) or 0.05
        return base * {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "4×": 4.0}[self.dd_frustum.value]

    def _draw_cameras(self) -> None:
        import viser.transforms as vtf

        d = self.dataset
        if d is None:
            return
        with self.cam_lock, self.server.atomic():
            for h in self.frustums:
                h.remove()
            self.frustums = []
            scale = self._frustum_scale()
            show = self.cb_cameras.value
            for i in range(len(d.names)):
                c2w = d.c2w[i]
                # nerfstudio/OpenGL camera -> viser/OpenCV camera: rotate pi about x
                R = vtf.SO3.from_matrix(c2w[:3, :3]) @ vtf.SO3.from_x_radians(np.pi)
                seq = int(d.sequence[i])
                visible = show and (self.seq_checkboxes[seq].value if seq in self.seq_checkboxes else True)
                h = self.server.scene.add_camera_frustum(
                    f"/cameras/{d.sequence_names[seq]}/{i:05d}",
                    fov=float(d.fov_y[i]), aspect=float(d.aspect[i]), scale=scale,
                    color=PALETTE[seq % len(PALETTE)], wxyz=R.wxyz, position=c2w[:3, 3], visible=visible,
                )
                h.on_click(self._jump_to)
                self.frustums.append(h)

    def _set_cameras_visible(self) -> None:
        d = self.dataset
        if d is None:
            return
        show = self.cb_cameras.value
        with self.cam_lock, self.server.atomic():
            for i, h in enumerate(self.frustums):
                seq = int(d.sequence[i])
                h.visible = show and (self.seq_checkboxes[seq].value if seq in self.seq_checkboxes else True)

    def _jump_to(self, event) -> None:
        with event.client.atomic():
            event.client.camera.position = event.target.position
            event.client.camera.wxyz = event.target.wxyz

    def _draw_points(self) -> None:
        d = self.dataset
        if d is None:
            return
        with self.cam_lock:
            if not self.cb_points.value:
                if self.points_handle is not None:
                    self.points_handle.visible = False
                return
            pts = d.points()
            if pts is None:
                self._set_status("_no seed cloud (points3d.ply) in the dataset_")
                return
            if self.points_handle is not None:
                self.points_handle.remove()
            self.points_handle = self.server.scene.add_point_cloud(
                "/seed_points", pts[0], pts[1], point_size=float(self.sl_point_size.value), point_shape="circle"
            )

    # ---- view

    def _update_home(self) -> None:
        if self.dataset is not None and len(self.dataset.names) > 2:
            pts = self.dataset.c2w[:, :3, 3]
        elif self.splat is not None:
            m = self.splat.means
            pts = m[np.random.default_rng(0).choice(len(m), min(len(m), 20000), replace=False)]
        else:
            return
        centre = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
        axis = vt[0]
        # start at one end of the principal axis looking down it, like walking into the module
        proj = (pts - centre) @ axis
        lo, hi = np.percentile(proj, [10, 90])
        eye = centre + axis * lo
        self.home = (eye.astype(float), (centre + axis * hi).astype(float))

    def _go_home(self, client) -> None:
        if self.home is None:
            return
        eye, look = self.home
        with client.atomic():
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.position = tuple(eye)
            client.camera.look_at = tuple(look)

    def _on_connect(self, client) -> None:
        self._go_home(client)

    # ---- main loop

    def run(self) -> None:
        log(self.source.describe())
        if self.source.run_dir is not None:
            log(f"run dir {short(self.source.run_dir)}")
        if self.args.share:
            self.server.request_share_url()
        self.check(force=True)
        if self.splat is None:
            if not self.cb_watch.value:
                raise SystemExit(f"nothing to show under {self.source.path} (no .ply / step-*.ckpt); "
                                 "use --watch to wait for training to write one")
            log("nothing to load yet; watching …")
        log(f"open http://localhost:{self.server.get_port()}")
        try:
            while True:
                time.sleep(self.args.poll_interval)
                if self.cb_watch.value:
                    self.check()
        except KeyboardInterrupt:
            pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="space-3dgs-viewer",
        description=__doc__.split("\n\n")[0],
        epilog="\n\n".join(__doc__.split("\n\n")[1:]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", help="splat .ply, a run's config.yml / .ckpt / directory, or an experiment dir "
                                 "(outputs/<exp>: newest run inside)")
    ap.add_argument("--data", default=None,
                    help="dataset (dir or transforms.json) for the camera overlay; default: the run's config.yml")
    ap.add_argument("--run", default=None,
                    help="training run dir whose dataparser_transforms.json / config.yml to use "
                         "(default: found next to the source)")
    ap.add_argument("--watch", action="store_true", help="keep polling the source and hot-swap new checkpoints")
    ap.add_argument("--poll-interval", type=float, default=10.0, help="seconds between checks (default 10)")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="ignore files modified less than this many seconds ago (still being written)")
    ap.add_argument("--max-gaussians", type=int, default=0,
                    help="cap the number of Gaussians sent to the browser, keeping the most opaque (0 = all)")
    ap.add_argument("--min-opacity", type=float, default=0.0, help="drop Gaussians below this opacity")
    ap.add_argument("--max-cameras", type=int, default=1000,
                    help="draw at most this many training frustums (evenly strided; 0 = all)")
    ap.add_argument("--frustum-scale", type=float, default=0.0,
                    help="frustum size in scene units (0 = 2%% of the camera bounding-box diagonal)")
    ap.add_argument("--no-cameras", action="store_true", help="start with the camera frustums hidden")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7007)
    ap.add_argument("--share", action="store_true", help="also request a public viser share URL")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if importlib.util.find_spec("viser") is None:
        sys.exit("viser is not installed (it comes with nerfstudio; or `pip install viser`)")
    Viewer(args).run()


if __name__ == "__main__":
    main()
