"""Convert a COLMAP sparse model into a nerfstudio dataset.

Reads cameras/images/points3D via pycolmap (bin or txt, auto-detected) and
writes  <out>/
    transforms.json    per-frame c2w in nerfstudio/OpenGL convention
                       (COLMAP stores OpenCV world-to-camera; we invert and
                       right-multiply by diag(1,-1,-1,1))
    images/<name>      symlinks into --images-root
    points3d.ply       the model's 3D points (seeds splatfacto init);
                       skipped with a warning if the model has none

Intrinsics are written top-level when the model has a single camera, and
per-frame when it has several.

--align-to <transforms.json>: estimate a Sim3 (scale, R, t; Umeyama on
camera centers of frames present in BOTH models, matched by filename
basename) that maps the COLMAP frame into the reference frame, and apply it
to all poses and points before writing. The post-alignment RMS camera-center
residual (meters, assuming the reference is metric) is reported overall and
per sequence (sequence = filename prefix before the last _<digits>); it
measures reference-vs-SfM pose disagreement.

Needs pycolmap. scripts/sfm_global.py calls this for you; run it directly to
convert any COLMAP model:
  python scripts/colmap_to_transforms.py \
      --model path/to/sparse/0 \
      --images-root path/to/images \
      --output data/my_scene \
      [--align-to data/my_scene_prior/transforms.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pycolmap

# OpenGL <-> OpenCV camera-axis flip (its own inverse).
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

SEQ_RE = re.compile(r"^(.*)_\d+$")


def seq_key(name: str) -> str:
    """Sequence id = filename prefix before the last _<digits>."""
    m = SEQ_RE.match(Path(name).stem)
    return m.group(1) if m else Path(name).stem


def cam_from_world_matrix(image) -> np.ndarray:
    """3x4 world-to-camera (OpenCV) matrix, across pycolmap API variants."""
    cfw = image.cam_from_world
    if callable(cfw):
        cfw = cfw()
    return np.asarray(cfw.matrix())


def intrinsics_dict(cam) -> dict:
    """COLMAP camera -> nerfstudio OPENCV intrinsics keys."""
    model = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
    p = list(cam.params)
    d = {"w": int(cam.width), "h": int(cam.height),
         "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    if model == "SIMPLE_PINHOLE":
        d.update(fl_x=p[0], fl_y=p[0], cx=p[1], cy=p[2])
    elif model == "PINHOLE":
        d.update(fl_x=p[0], fl_y=p[1], cx=p[2], cy=p[3])
    elif model == "SIMPLE_RADIAL":
        d.update(fl_x=p[0], fl_y=p[0], cx=p[1], cy=p[2], k1=p[3])
    elif model == "RADIAL":
        d.update(fl_x=p[0], fl_y=p[0], cx=p[1], cy=p[2], k1=p[3], k2=p[4])
    elif model == "OPENCV":
        d.update(fl_x=p[0], fl_y=p[1], cx=p[2], cy=p[3],
                 k1=p[4], k2=p[5], p1=p[6], p2=p[7])
    else:
        raise SystemExit(f"unsupported COLMAP camera model: {model}")
    return d


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (s, R, t) minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="COLMAP sparse model dir (bin or txt)")
    ap.add_argument("--images-root", required=True, help="dir holding the source images")
    ap.add_argument("--output", required=True, help="output nerfstudio dataset dir")
    ap.add_argument("--align-to", default=None,
                    help="reference transforms.json; Sim3-align poses+points to it")
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(str(Path(args.model).resolve()))
    images_root = Path(args.images_root).resolve()
    out = Path(args.output).resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    print(f"[model] {rec.num_images()} images, {len(rec.cameras)} cameras, "
          f"{rec.num_points3D()} points3D")

    # --- poses: COLMAP OpenCV w2c -> OpenGL c2w, keyed/sorted by image name.
    frames = []  # (name, camera_id, c2w_gl 4x4)
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        M = cam_from_world_matrix(im)          # 3x4 w2c (OpenCV)
        R_wc, t = M[:3, :3], M[:3, 3]
        c2w = np.eye(4)
        c2w[:3, :3] = R_wc.T
        c2w[:3, 3] = -R_wc.T @ t
        frames.append([im.name, im.camera_id, c2w @ FLIP])

    # --- optional Sim3 alignment to a reference transforms.json.
    if args.align_to:
        ref = json.loads(Path(args.align_to).read_text())
        ref_centers = {Path(fr["file_path"]).name: np.array(fr["transform_matrix"])[:3, 3]
                       for fr in ref["frames"]}
        names, src_c, dst_c = [], [], []
        for name, _, c2w in frames:
            key = Path(name).name
            if key in ref_centers:
                names.append(key)
                src_c.append(c2w[:3, 3])
                dst_c.append(ref_centers[key])
        if len(names) < 3:
            raise SystemExit(f"--align-to: only {len(names)} common frames, need >=3")
        src_c, dst_c = np.asarray(src_c), np.asarray(dst_c)
        s, R, t = umeyama(src_c, dst_c)
        resid = dst_c - (s * (R @ src_c.T).T + t)
        err = np.linalg.norm(resid, axis=1)
        print(f"[align] {len(names)} common frames, scale {s:.6f}")
        print(f"[align] camera-center residual after Sim3: "
              f"RMS {np.sqrt((err**2).mean()):.4f} m, "
              f"mean {err.mean():.4f} m, max {err.max():.4f} m")
        by_seq = {}
        for n, e in zip(names, err):
            by_seq.setdefault(seq_key(n), []).append(e)
        for sk in sorted(by_seq):
            e = np.asarray(by_seq[sk])
            print(f"[align]   {sk}: n={len(e)} RMS {np.sqrt((e**2).mean()):.4f} m "
                  f"max {e.max():.4f} m")
        for fr in frames:
            c2w = fr[2]
            new = np.eye(4)
            new[:3, :3] = R @ c2w[:3, :3]
            new[:3, 3] = s * (R @ c2w[:3, 3]) + t
            fr[2] = new
    else:
        s, R, t = 1.0, np.eye(3), np.zeros(3)

    # --- intrinsics (top-level if a single camera, else per-frame).
    cam_intr = {cid: intrinsics_dict(cam) for cid, cam in rec.cameras.items()}
    single = len(cam_intr) == 1

    frames_json, n_missing = [], 0
    for name, cid, c2w in frames:
        base = Path(name).name
        src_img = images_root / name
        if not src_img.exists():
            src_img = images_root / base
        if not src_img.exists():
            n_missing += 1
            continue
        link = out / "images" / base
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src_img.resolve())
        fr = {"file_path": f"images/{base}",
              "transform_matrix": c2w.tolist()}
        if not single:
            fr.update(cam_intr[cid])
        frames_json.append(fr)
    if n_missing:
        print(f"[warn] {n_missing} model images not found under {images_root} (skipped)")

    meta = {"camera_model": "OPENCV"}
    if single:
        meta.update(next(iter(cam_intr.values())))
    meta["orientation_override"] = "none"
    meta["frames"] = frames_json

    # --- points3d.ply from the model's 3D points (Sim3-transformed if aligned).
    if rec.num_points3D() > 0:
        xyz = np.array([p.xyz for p in rec.points3D.values()])
        rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
        xyz = s * (R @ xyz.T).T + t
        write_ply(out / "points3d.ply", xyz.astype(np.float32), rgb)
        meta["ply_file_path"] = "points3d.ply"
        print(f"[ply] {len(xyz)} points -> points3d.ply, "
              f"bbox min {xyz.min(0).round(2)} max {xyz.max(0).round(2)}")
    else:
        print("[warn] model has zero points3D; skipping points3d.ply "
              "(splatfacto will fall back to random init)")

    (out / "transforms.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {len(frames_json)} frames -> {out / 'transforms.json'}")


if __name__ == "__main__":
    main()
