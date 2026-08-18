"""Loaders for the Astrobee ISS dataset (TUM RGB-D layout).

A sequence directory contains:
  gray/<ts>.png            undistorted 1-channel-replicated PNGs (1280x880 for the
                           pinhole undistorted_calib below)
  gray.txt                 lines:  <ts> gray/<ts>.png
  groundtruth.txt          lines:  <ts> tx ty tz qx qy qz qw   (ISS world frame)
  undistorted_calib.txt    fx fy cx cy               (pinhole, for gray/)
  distorted_calib.txt      fx fy cx cy w             (FOV model, for gray_raw/)
  description.yaml, imu.txt, ...

Images and poses are aligned 1:1 by identical timestamp (verified on iva_kibo_trans:
1068 images, 1068 poses, matching timestamps), so no TUM association step is needed;
we still intersect on timestamp for robustness.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Pinhole:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def load_undistorted_calib(seq_dir: Path) -> Pinhole:
    vals = [float(x) for x in (seq_dir / "undistorted_calib.txt").read_text().split()]
    fx, fy, cx, cy = vals[:4]
    # Image dimensions: principal point is at the image centre for the undistorted
    # rectification, so width = 2*cx, height = 2*cy (1280x880 on iva_kibo_trans).
    return Pinhole(fx, fy, cx, cy, width=int(round(2 * cx)), height=int(round(2 * cy)))


def load_image_index(seq_dir: Path) -> "dict[int, str]":
    """timestamp (int ns) -> relative image path, from gray.txt."""
    out = {}
    for line in (seq_dir / "gray.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ts, rel = line.split()
        out[int(ts)] = rel
    return out


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """TUM quaternion (x,y,z,w) -> 3x3 rotation matrix. Normalises first."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_poses(seq_dir: Path) -> "dict[int, np.ndarray]":
    """timestamp (int ns) -> 4x4 pose matrix T (world <- camera-frame-as-given).

    Each row of groundtruth.txt is `ts tx ty tz qx qy qz qw`. We interpret the
    (t, q) as the pose of the camera expressed in the ISS world frame, i.e. the
    matrix maps a point in the pose's local frame to world coordinates:
        p_world = R * p_local + t.
    The exact optical-axis convention of that local frame (OpenCV vs OpenGL) is
    resolved by build_c2w() via the `flip` argument.
    """
    out = {}
    for line in (seq_dir / "groundtruth.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        ts = int(p[0])
        tx, ty, tz, qx, qy, qz, qw = map(float, p[1:8])
        T = np.eye(4)
        T[:3, :3] = quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = (tx, ty, tz)
        out[ts] = T
    return out


# Axis-convention flips applied on the RIGHT of the given pose to change the
# camera's local optical-frame axes without moving the camera centre.
#   opencv : x-right, y-down,  z-forward (into scene)  -- ROS optical frame
#   opengl : x-right, y-up,    z-back                  -- nerfstudio / OpenGL
# opencv -> opengl : negate the y and z axes.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def build_c2w(T_given: np.ndarray, source_convention: str = "opencv") -> np.ndarray:
    """Return a camera-to-world matrix in nerfstudio (OpenGL) convention.

    T_given maps local-camera-frame points to world. If that local frame is the
    OpenCV optical frame, right-multiply by the CV->GL flip so the resulting
    matrix's columns are (right, up, back) in world space.
    """
    if source_convention == "opencv":
        return T_given @ _CV_TO_GL
    elif source_convention == "opengl":
        return T_given.copy()
    else:
        raise ValueError(source_convention)


def paired_frames(seq_dir: Path):
    """Return sorted list of (ts, image_relpath, T_given 4x4) for frames that have
    both an image and a pose."""
    imgs = load_image_index(seq_dir)
    poses = load_poses(seq_dir)
    common = sorted(set(imgs) & set(poses))
    return [(ts, imgs[ts], poses[ts]) for ts in common]
