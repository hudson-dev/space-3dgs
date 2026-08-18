"""Download Astrobee ISS sequences (or CAD models) by name via gdown.

Google-Drive file IDs come from manifest.json next to this script (regenerate
it with parse_manifest.py if the dataset page changes).

Usage:
  python examples/iss_jem/download.py seq ff_return_journey_forward ff_return_journey_up
  python examples/iss_jem/download.py cad JEM
Downloads land in <repo>/data/raw/<name>/ (sequences) or <repo>/data/cad/<name>/,
extracted if archives. Re-running skips complete downloads.
"""
import json
import sys
import zipfile
import tarfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((Path(__file__).with_name("manifest.json")).read_text())


def _missing_frames(dest_dir: Path) -> "list[str] | None":
    """Relative image paths listed in gray.txt that are absent from disk.

    None means gray.txt itself is missing. An unextracted gray.zip counts as
    complete, since prepare_sequence.py expands it afterwards.
    """
    index = dest_dir / "gray.txt"
    if not index.exists():
        return None
    if (dest_dir / "gray.zip").exists():
        return []
    rels = [
        line.split()[-1]
        for line in index.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not rels:
        return None
    return [rel for rel in rels if not (dest_dir / rel).exists()]


def _payload_present(dest_dir: Path, tag: str) -> bool:
    """True if the download marker's expected payload is still on disk."""
    if tag.startswith("seq:"):
        missing = _missing_frames(dest_dir)
        if missing is None:
            return False
        if missing:
            print(f"[warn] {tag}: {len(missing)} frames from gray.txt missing "
                  f"(e.g. {missing[0]})")
            return False
        return True
    return any(p.name != ".download_complete" for p in dest_dir.iterdir())


def fetch(file_id: str, dest_dir: Path, tag: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    done = dest_dir / ".download_complete"
    if done.exists():
        if _payload_present(dest_dir, tag):
            print(f"[skip] {tag} already downloaded in {dest_dir}")
            return
        done.unlink()
        print(f"[redo] {tag} marker present but data missing; re-downloading")
    # gdown writes the file with its Drive name; use fuzzy + output dir.
    print(f"[get ] {tag}  id={file_id}  -> {dest_dir}")
    out = gdown.download(id=file_id, output=str(dest_dir) + "/", quiet=False, fuzzy=True)
    if out is None:
        raise SystemExit(f"[FAIL] {tag} download returned None")
    out = Path(out)
    print(f"[ok  ] downloaded {out} ({out.stat().st_size/1e6:.1f} MB)")
    # Auto-extract archives.
    if out.suffix == ".zip" or zipfile.is_zipfile(out):
        print(f"[unzip] {out.name}")
        with zipfile.ZipFile(out) as z:
            z.extractall(dest_dir)
    elif out.suffix in {".tar", ".gz", ".tgz", ".bz2"} or tarfile.is_tarfile(out):
        print(f"[untar] {out.name}")
        with tarfile.open(out) as t:
            t.extractall(dest_dir)
    done.write_text("ok\n")
    print(f"[done] {tag}")


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else ""
    names = sys.argv[2:]
    if kind not in {"seq", "cad"}:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} {{seq|cad}} NAME [NAME...]")
    if not names:
        key = "sequences" if kind == "seq" else "cad"
        raise SystemExit(f"no {kind} names given; available: "
                         f"{', '.join(sorted(MANIFEST[key]))}")
    for name in names:
        if kind == "seq":
            info = MANIFEST["sequences"][name]
            fetch(info["processed_id"], ROOT / "data" / "raw" / name, f"seq:{name}")
        else:
            fetch(MANIFEST["cad"][name], ROOT / "data" / "cad" / name, f"cad:{name}")


if __name__ == "__main__":
    main()
