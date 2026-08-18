"""Post-download prep for a sequence directory:
  - extract the nested gray.zip -> gray/<ts>.png (undistorted images)
  - delete redundant large archives (outer <name>.zip, gray.zip, gray_raw.zip) to
    reclaim disk; we keep only gray/ plus the small text files
Idempotent: safe to re-run.

Usage: python examples/iss_jem/prepare_sequence.py data/raw/<name> [more dirs...]
"""
import sys
import zipfile
from pathlib import Path


def _png_count(gray: Path) -> int:
    return len(list(gray.glob("*.png"))) if gray.exists() else 0


def _expected_png_count(seq_dir: Path, gz: Path):
    """How many PNGs a complete extract should have (gray.txt, else zip)."""
    gt = seq_dir / "gray.txt"
    if gt.exists():
        n = 0
        for line in gt.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
        return n
    if gz.exists():
        with zipfile.ZipFile(gz) as z:
            return sum(1 for name in z.namelist() if name.lower().endswith(".png"))
    return None


def prepare(seq_dir: Path):
    seq_dir = Path(seq_dir)
    gray = seq_dir / "gray"
    gz = seq_dir / "gray.zip"
    expected = _expected_png_count(seq_dir, gz)
    have = _png_count(gray)
    complete = expected is not None and have >= expected

    if gz.exists() and not complete:
        gray.mkdir(exist_ok=True)
        with zipfile.ZipFile(gz) as z:
            names = z.namelist()
            # entries may be flat "<ts>.png" or "gray/<ts>.png"
            for n in names:
                if not n.lower().endswith(".png"):
                    continue
                data = z.read(n)
                (gray / Path(n).name).write_bytes(data)
        have = _png_count(gray)
        expected = _expected_png_count(seq_dir, gz)
        complete = expected is not None and have >= expected
        print(f"[extract] {have} pngs -> {gray}")

    # reclaim disk; keep gray.zip until the extract is verified complete
    for junk in [seq_dir / f"{seq_dir.name}.zip", gz, seq_dir / "gray_raw.zip"]:
        if junk == gz and not complete:
            continue
        if junk.exists():
            mb = junk.stat().st_size / 1e6
            junk.unlink()
            print(f"[rm] {junk.name} ({mb:.0f} MB)")

    n = _png_count(gray)
    print(f"[ok] {seq_dir.name}: {n} images in gray/")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        prepare(Path(d))
