"""Prior-guided global structure-from-motion -> nerfstudio dataset.

Input : a nerfstudio dataset (transforms.json + images/) whose poses are only
        approximate — robot localization, odometry, a previous solve. Images
        must share one set of intrinsics (top-level fl_x/fl_y/cx/cy in
        transforms.json) and be listed as images/<name>.
Output: a new nerfstudio dataset with the re-solved poses, the SfM point cloud
        (points3d.ply, the splat initialisation) and the COLMAP model, all
        registered into the input's frame by a robust Sim(3) fit, plus
        sfm_report.json with per-stage timings and registration statistics.

Stages (each is skipped when its marker exists, so an interrupted run resumes):
  1. SIFT extraction, one shared camera from the input intrinsics
  2. sequential matching over the name-sorted image list
  3. position priors from the input poses written into the database
  4. spatial matching: pairs proposed among images whose prior positions lie
     within --spatial-max-distance (nearest --spatial-neighbors), so images
     from different passes/directions that see the same surfaces get matched
     even when the sequential chain never links them
  5. GLOMAP global mapping (rotation averaging -> global positioning -> BA)
  6. Sim(3) registration of the largest model onto the prior camera centres
     (RANSAC with --align-max-error inlier threshold, metres if the priors are)
  7. dataset export via colmap_to_transforms.py

The priors only decide which image pairs are attempted (4) and fix the metric
gauge (6); every match is verified geometrically and the prior poses never
enter the solve.

Needs pycolmap >= 3.11 (GLOMAP is bundled). Multi-hour for thousands of
images on CPU; use nohup/tmux.
  python scripts/sfm_global.py --data data/my_scene_prior --output data/my_scene
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="input nerfstudio dataset with prior poses")
    ap.add_argument("--output", required=True, help="output nerfstudio dataset dir")
    ap.add_argument("--work", default=None,
                    help="scratch dir for the feature database and raw models "
                         "(default <output>/colmap/work; put it on a fast disk)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the scratch dir (database.db, unaligned models)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="SIFT extraction/matching device; auto = cuda if this "
                         "pycolmap build has it, else cpu (mapping is always CPU)")
    ap.add_argument("--seq-overlap", type=int, default=10,
                    help="sequential matching: neighbours per image")
    ap.add_argument("--spatial-neighbors", type=int, default=30,
                    help="spatial matching: nearest prior positions per image")
    ap.add_argument("--spatial-max-distance", type=float, default=3.0,
                    help="spatial matching: max prior-centre distance (input units)")
    ap.add_argument("--min-num-matches", type=int, default=30,
                    help="GLOMAP: min verified matches for a view-graph edge")
    ap.add_argument("--keep-max-tracks", type=int, default=0,
                    help="GLOMAP: cap on tracks kept for global positioning "
                         "(0 = uncapped; set e.g. 1500000 if it runs out of RAM)")
    ap.add_argument("--align-max-error", type=float, default=0.10,
                    help="Sim(3) RANSAC inlier threshold on camera centres")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work).resolve() if args.work else out / "colmap" / "work"
    work.mkdir(parents=True, exist_ok=True)
    model_dir = out / "colmap" / "sparse" / "0"
    report_path = out / "sfm_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["pycolmap_version"] = pycolmap.__version__

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2))

    # --- input dataset: shared intrinsics, image names, prior camera centres.
    meta = json.loads((data / "transforms.json").read_text())
    for k in ("fl_x", "fl_y", "cx", "cy"):
        if k not in meta:
            raise SystemExit(f"{data}/transforms.json needs top-level {k} (shared intrinsics)")
    if meta.get("camera_model", "OPENCV") not in ("OPENCV", "PINHOLE"):
        raise SystemExit(f"unsupported camera_model {meta['camera_model']} (need OPENCV/PINHOLE)")
    dist = [float(meta.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2")]
    if any(dist):
        cam_model = "OPENCV"
        cam_params = ",".join(str(v) for v in
                              [meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"], *dist])
    else:
        cam_model = "PINHOLE"
        cam_params = ",".join(str(meta[k]) for k in ("fl_x", "fl_y", "cx", "cy"))

    images_root = data / "images"
    names, prior_center = [], {}
    for fr in meta["frames"]:
        fp = Path(fr["file_path"])
        if fp.parts[0] != "images":
            raise SystemExit(f"frame path {fp} is not under images/")
        name = str(Path(*fp.parts[1:]))
        if not (images_root / name).exists():
            raise SystemExit(f"missing image {images_root / name}")
        names.append(name)
        prior_center[name] = np.asarray(fr["transform_matrix"], dtype=np.float64)[:3, 3]
    names = sorted(names)
    if len(names) != len(set(names)):
        raise SystemExit("duplicate image names in transforms.json")
    report.update(n_images=len(names), camera_model=cam_model)
    print(f"[input] {len(names)} images, {cam_model} {cam_params}", flush=True)
    save()

    db_path = work / "database.db"
    sparse = work / "sparse"
    use_gpu = args.device == "cuda" or (args.device == "auto" and bool(getattr(pycolmap, "has_cuda", False)))
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    report["sift_device"] = "cuda" if use_gpu else "cpu"

    def stage(marker: str, key: str, fn) -> None:
        m = work / marker
        if model_dir.exists() or m.exists():
            print(f"[{key}] artifact exists, skipping", flush=True)
            return
        t0 = time.time()
        fn()
        report[key] = round(time.time() - t0, 1)
        m.touch()
        save()
        print(f"[{key}] done in {report[key]}s", flush=True)

    # --- 1. SIFT extraction, one shared camera.
    def extract():
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = cam_model
        reader.camera_params = cam_params
        pycolmap.extract_features(
            database_path=str(db_path), image_path=str(images_root),
            image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader, device=device)
    stage(".extracted", "extract_s", extract)

    # --- 2. Sequential matching over the name-sorted chain.
    def match_seq():
        fm = pycolmap.FeatureMatchingOptions()
        fm.use_gpu = use_gpu
        opts = pycolmap.SequentialPairingOptions()
        opts.overlap = args.seq_overlap
        opts.quadratic_overlap = False
        opts.loop_detection = False
        pycolmap.match_sequential(database_path=str(db_path),
                                  matching_options=fm, pairing_options=opts)
    stage(".seq_matched", "match_sequential_s", match_seq)

    # --- 3. Position priors from the input poses.
    def priors():
        db = pycolmap.Database.open(str(db_path))
        if db.num_pose_priors() == 0:
            n = 0
            for im in db.read_all_images():
                pp = pycolmap.PosePrior()
                pp.position = prior_center[im.name]
                pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
                pp.corr_data_id = im.data_id
                db.write_pose_prior(pp)
                n += 1
            report["priors_written"] = n
        db.close()
    stage(".priors", "priors_s", priors)

    # --- 4. Spatial matching from the priors.
    def match_spatial():
        fm = pycolmap.FeatureMatchingOptions()
        fm.use_gpu = use_gpu
        sp = pycolmap.SpatialPairingOptions()
        sp.ignore_z = False
        sp.max_num_neighbors = args.spatial_neighbors
        sp.max_distance = args.spatial_max_distance
        pycolmap.match_spatial(database_path=str(db_path),
                               matching_options=fm, pairing_options=sp)
    stage(".spatial_matched", "match_spatial_s", match_spatial)

    # --- 5. GLOMAP global mapping (CPU).
    def glomap():
        go = pycolmap.GlobalPipelineOptions()
        go.min_num_matches = args.min_num_matches
        go.mapper.global_positioning.use_gpu = False
        go.mapper.bundle_adjustment.ceres.use_gpu = False
        if args.keep_max_tracks > 0:
            go.mapper.keep_max_num_tracks = args.keep_max_tracks
            report["keep_max_num_tracks"] = args.keep_max_tracks
        recs = pycolmap.global_mapping(database_path=str(db_path),
                                       image_path=str(images_root),
                                       output_path=str(sparse), options=go)
        report["models"] = [
            {"model_index": idx,
             "num_reg_images": rec.num_reg_images(),
             "num_points3D": rec.num_points3D(),
             "mean_track_length": round(rec.compute_mean_track_length(), 3),
             "mean_reproj_error_px": round(rec.compute_mean_reprojection_error(), 3)}
            for idx, rec in recs.items()]
        print(f"[glomap] models: {report['models']}", flush=True)
    stage(".mapped", "global_mapping_s", glomap)

    # --- 6. Sim(3)-register the largest model onto the prior camera centres.
    if not model_dir.exists():
        t0 = time.time()
        models = sorted(d for d in sparse.iterdir() if d.is_dir() and d.name.isdigit())
        if not models:
            raise SystemExit("GLOMAP produced no model")
        best = max((pycolmap.Reconstruction(str(d)) for d in models),
                   key=lambda r: r.num_reg_images())
        tgt_names, tgt_locs = [], []
        for im in best.images.values():
            if im.has_pose and im.name in prior_center:
                tgt_names.append(im.name)
                tgt_locs.append(prior_center[im.name])
        sim3 = pycolmap.align_reconstruction_to_locations(
            best, tgt_names, np.asarray(tgt_locs), min_common_images=3,
            ransac_options=pycolmap.RANSACOptions(max_error=args.align_max_error))
        if sim3 is None:
            raise SystemExit("Sim(3) registration to the prior camera centres failed")
        best.transform(sim3)
        errs = np.array([np.linalg.norm(best.image(i).projection_center()
                                        - prior_center[best.image(i).name])
                         for i in best.reg_image_ids()])
        report["align"] = {
            "n_images": int(len(errs)),
            "unregistered": int(len(names) - len(errs)),
            "median_dev": round(float(np.median(errs)), 4),
            "p95_dev": round(float(np.percentile(errs, 95)), 4),
            "max_dev": round(float(errs.max()), 4)}
        report["align_s"] = round(time.time() - t0, 1)
        model_dir.mkdir(parents=True)
        best.write(str(model_dir))
        save()
        print(f"[align] registered {len(errs)}/{len(names)} images; "
              f"centre deviation vs priors: {report['align']}", flush=True)
    else:
        print("[align] colmap/sparse/0 exists, skipping", flush=True)

    # --- 7. nerfstudio dataset.
    if not (out / "transforms.json").exists():
        t0 = time.time()
        subprocess.run(
            [sys.executable, str(HERE / "colmap_to_transforms.py"),
             "--model", str(model_dir), "--images-root", str(images_root),
             "--output", str(out)], check=True)
        report["dataset_s"] = round(time.time() - t0, 1)
        save()
    else:
        print("[dataset] transforms.json exists, skipping", flush=True)

    report["sfm_total_s"] = round(sum(
        report.get(k, 0.0) for k in ("extract_s", "match_sequential_s", "priors_s",
                                     "match_spatial_s", "global_mapping_s", "align_s")), 1)
    save()
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[done] {out}  (SfM {report['sfm_total_s'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
