"""Convert one or more Astrobee TUM sequences into a nerfstudio dataset.

Produces  <out>/
    transforms.json           (fl_x/fl_y/cx/cy/w/h, OPENCV model, per-frame c2w in OpenGL)
    images/<seq>_<ts>.png     (symlinks to the undistorted gray PNGs)
    points3d.ply              (frustum-unprojected seed cloud, in world coords)

The dataset's localization poses are camera-to-world in the OpenCV optical
frame (verified by an epipolar Sampson test). nerfstudio wants OpenGL c2w, so
each pose is right-multiplied by diag(1,-1,-1,1).

Near-duplicate frames (5 Hz on a slowly moving robot) are dropped by a
motion threshold: a frame is kept only if it moved > trans_thresh metres OR
rotated > rot_thresh degrees since the last kept frame.

The resulting dataset is the *prior-pose* input to scripts/sfm_global.py; the
seed cloud is only useful if you train on these poses directly, so the demo
passes --no-ply.

Usage:
  python examples/iss_jem/convert_to_nerfstudio.py \
      --out data/iss_jem_prior \
      --seqs data/raw/ff_return_journey_{forward,up,down,left,right} \
      --trans-thresh 0.015 --rot-thresh 0.75 --no-ply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from astrobee_io import (  # noqa: E402
    paired_frames, load_undistorted_calib, build_c2w,
)


def rot_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def dedup(frames, trans_thresh, rot_thresh):
    """frames: list of (ts, relpath, T_given). Keep by motion threshold."""
    kept = []
    last_C = None
    last_R = None
    for ts, rel, T in frames:
        C = T[:3, 3]
        R = T[:3, :3]
        if last_C is None:
            kept.append((ts, rel, T))
            last_C, last_R = C, R
            continue
        dt = np.linalg.norm(C - last_C)
        dr = rot_angle_deg(last_R, R)
        if dt > trans_thresh or dr > rot_thresh:
            kept.append((ts, rel, T))
            last_C, last_R = C, R
    return kept


def seed_cloud(kept_by_seq, cal, every=5, grid=(48, 33),
               d_min=0.35, d_max=3.0, max_points=250_000, seed=0):
    """Unproject a coarse pixel grid at randomised depth for a subset of frames,
    giving points inside the camera frustums (near the real surfaces). Returns
    (xyz Nx3 world, rgb Nx3 uint8)."""
    import cv2  # only the seed cloud needs OpenCV

    rng = np.random.default_rng(seed)
    K = np.array([[cal.fx, 0, cal.cx], [0, cal.fy, cal.cy], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)
    us = np.linspace(0.06 * cal.width, 0.94 * cal.width, grid[0])
    vs = np.linspace(0.06 * cal.height, 0.94 * cal.height, grid[1])
    uu, vv = np.meshgrid(us, vs)
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 1)  # G x 3
    dirs_cam = (Kinv @ pix.T).T                                    # G x 3 (OpenCV, +z fwd)

    all_xyz, all_rgb = [], []
    for seq_dir, kept in kept_by_seq:
        for k in range(0, len(kept), every):
            ts, rel, T = kept[k]  # T is OpenCV c2w
            img = cv2.imread(str(seq_dir / rel), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            d = rng.uniform(d_min, d_max, size=dirs_cam.shape[0])[:, None]
            pts_cam = dirs_cam * d
            pts_w = (T[:3, :3] @ pts_cam.T + T[:3, 3:4]).T
            ui = np.clip(pix[:, 0].astype(int), 0, cal.width - 1)
            vi = np.clip(pix[:, 1].astype(int), 0, cal.height - 1)
            g = img[vi, ui]
            all_xyz.append(pts_w)
            all_rgb.append(np.stack([g, g, g], 1))
    if not all_xyz:
        raise RuntimeError(
            "seed_cloud: no images could be loaded; ensure gray/ PNGs exist "
            "(run prepare_sequence.py after download)"
        )
    xyz = np.concatenate(all_xyz)
    rgb = np.concatenate(all_rgb)
    if len(xyz) > max_points:
        sel = rng.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return xyz.astype(np.float32), rgb.astype(np.uint8)


def write_ply(path, xyz, rgb):
    n = len(xyz)
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode())
        arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--trans-thresh", type=float, default=0.015)
    ap.add_argument("--rot-thresh", type=float, default=0.75)
    ap.add_argument("--seed-every", type=int, default=5)
    ap.add_argument("--no-ply", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    cal0 = None
    frames_json = []
    kept_by_seq = []
    for sd in args.seqs:
        seq_dir = Path(sd).resolve()
        cal = load_undistorted_calib(seq_dir)
        if cal0 is None:
            cal0 = cal
        else:
            assert abs(cal.fx - cal0.fx) < 1e-3 and cal.width == cal0.width, \
                f"intrinsics differ for {seq_dir}"
        frames = [
            (ts, rel, T) for ts, rel, T in paired_frames(seq_dir)
            if (seq_dir / rel).exists()
        ]
        if not frames:
            print(f"[{seq_dir.name}] no image files on disk; skipping")
            continue
        kept = dedup(frames, args.trans_thresh, args.rot_thresh)
        kept_by_seq.append((seq_dir, kept))
        print(f"[{seq_dir.name}] {len(frames)} -> {len(kept)} kept "
              f"({100*len(kept)/len(frames):.0f}%)")
        for ts, rel, T in kept:
            name = f"{seq_dir.name}_{ts}.png"
            link = out / "images" / name
            if not link.exists():
                link.symlink_to(seq_dir / rel)
            c2w_gl = build_c2w(T, "opencv")
            frames_json.append({
                "file_path": f"images/{name}",
                "transform_matrix": c2w_gl.tolist(),
            })

    if not frames_json:
        raise SystemExit(
            "no frames available across the given sequences; run "
            "scripts/prepare_sequence.py to extract gray/ PNGs before converting"
        )

    meta = {
        "camera_model": "OPENCV",
        "fl_x": cal0.fx, "fl_y": cal0.fy, "cx": cal0.cx, "cy": cal0.cy,
        "w": cal0.width, "h": cal0.height,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "orientation_override": "none",
        "frames": frames_json,
    }

    if not args.no_ply:
        xyz, rgb = seed_cloud(kept_by_seq, cal0, every=args.seed_every)
        write_ply(out / "points3d.ply", xyz, rgb)
        meta["ply_file_path"] = "points3d.ply"
        print(f"[seed] wrote points3d.ply with {len(xyz)} points, "
              f"bbox min {xyz.min(0).round(2)} max {xyz.max(0).round(2)}")

    (out / "transforms.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {len(frames_json)} frames -> {out/'transforms.json'}")


if __name__ == "__main__":
    main()
