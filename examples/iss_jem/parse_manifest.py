"""Parse the Astrobee ISS dataset page into a clean download manifest.

Each sequence row is: <name>, <duration_s>, <processed download link>, <original bag link>.
Also captures the CAD model links (JEM / NOD2 / USL).

Usage:
  python examples/iss_jem/parse_manifest.py [astrobee_page.html]

If no HTML path is given, the dataset homepage is fetched.

Output: examples/iss_jem/manifest.json (next to this script)
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

DATASET_PAGE = "https://astrobee-iss-dataset.github.io/"


def load_html() -> str:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).read_text(errors="ignore")
    print(f"[get ] {DATASET_PAGE}")
    with urllib.request.urlopen(DATASET_PAGE) as resp:
        return resp.read().decode("utf-8", errors="ignore")


HTML = load_html()

DRIVE = r'https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)'

# Split into <tr>...</tr> blocks and pull out name + all drive IDs in order.
rows = re.findall(r"<tr>(.*?)</tr>", HTML, flags=re.DOTALL)

sequences = {}
name_re = re.compile(r"<td>\s*(?:&nbsp;)?\s*([A-Za-z0-9_]+)\s*</td>")
for row in rows:
    nm = name_re.search(row)
    if not nm:
        continue
    name = nm.group(1)
    if not re.match(r"^(iva_|ff_|td_|cal_)", name):
        continue
    ids = re.findall(DRIVE, row)
    # duration is the first plain numeric <td>
    dur = re.search(r'text-align:center">\s*(\d+)\s*</td>', row)
    sequences[name] = {
        "duration_s": int(dur.group(1)) if dur else None,
        "processed_id": ids[0] if len(ids) >= 1 else None,
        "bag_id": ids[1] if len(ids) >= 2 else None,
    }

# CAD models — known IDs from the page (labelled JEM / NOD2 / USL).
cad = {
    "JEM":  "1Uw4IJS2z0hgmO0VW28NydeckSkwRtiTD",
    "NOD2": "1L_PvXu3un-lMX22FJ9Gi3dT6DCcIOSev",
    "USL":  "1oQNjwUwvV5fB5NF7AVbBVbrKxKZ90uRh",
}

manifest = {"sequences": sequences, "cad": cad}
out = Path(__file__).with_name("manifest.json")
out.write_text(json.dumps(manifest, indent=2))

print(f"Parsed {len(sequences)} sequences -> {out}")
for n, v in sequences.items():
    flag = "" if v["processed_id"] else "  <-- MISSING processed_id"
    print(f"  {n:32s} {str(v['duration_s']):>5}s  {v['processed_id']}{flag}")
print("CAD:", cad)
