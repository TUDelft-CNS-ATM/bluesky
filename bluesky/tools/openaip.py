"""
Extract OpenAIP data from their public Google Cloud Storage bucket.

Downloads JSON exports for all countries and merges them into one
combined JSON file per dataset type. Results are saved in
'bluesky/resources/navdata/' alongside BlueSky's built-in navigation data.

Available dataset types:
    airports          Aerodromes and heliports
    airspaces         Controlled and restricted airspace
    navaids           Radio navigation aids (VOR, NDB, etc.)
    hotspots          Thermal hotspots for gliders
    obstacles         Obstacles (towers, cranes, etc.)
    reporting_points  VFR reporting points
    hang_gliding_sites Hang gliding and paragliding sites

Usage:
    python openaip.py                        # download all
    python openaip.py airspaces              # single dataset
    python openaip.py airspaces airports navaids
"""

import argparse
import gzip
import json
import os
import re
import time
import urllib.request
import urllib.error

# ── Configuration ─────────────────────────────────────────────────────────────
BUCKET_ID   = "29f98e10-a489-4c82-ae5e-489dbcd4912f"
LIST_URL    = f"https://storage.googleapis.com/storage/v1/b/{BUCKET_ID}/o"
DOWNLOAD_URL = f"https://storage.googleapis.com/{BUCKET_ID}"
# Data is stored alongside BlueSky's built-in navdata so plugins can reference
# it with a stable __file__-relative path.
OUTPUT_DIR  = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "navdata")
)

# Maps OpenAIP short code → human-readable output filename (without .json)
DATA_TYPES = {
    "apt": "airports",
    "asp": "airspaces",
    "nav": "navaids",
    "hot": "hotspots",
    "obs": "obstacles",
    "rpp": "reporting_points",
    "hgl": "hang_gliding_sites",
}
# Reverse lookup:  name → short code
NAME_TO_CODE = {v: k for k, v in DATA_TYPES.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_json(url):
    """Download a URL and return parsed JSON, or None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"  ⚠  {url}: {exc}")
        return None


def list_bucket_objects():
    """Return all objects in the GCS bucket (handles pagination)."""
    items, page_token = [], None
    while True:
        url = f"{LIST_URL}?maxResults=1000" + (f"&pageToken={page_token}" if page_token else "")
        data = fetch_json(url)
        if data is None:
            break
        items.extend(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return items


def find_country_files(objects):
    """
    Scan the object list for files matching {cc}_{type}.json.
    Returns a dict {dtype: [country_code, ...]} sorted by country.
    """
    pattern = re.compile(r"^([a-z]{2})_([a-z]{3})\.json$")
    type_to_countries = {}
    for obj in objects:
        m = pattern.match(obj["name"])
        if m:
            cc, dtype = m.group(1), m.group(2)
            type_to_countries.setdefault(dtype, []).append(cc)
    return {dtype: sorted(ccs) for dtype, ccs in type_to_countries.items()}


def download_and_merge(dtype, countries):
    """
    Download all per-country files for a data type and return a merged list.
    Each record is tagged with a '_country' field.
    """
    merged = []
    for cc in countries:
        filename = f"{cc}_{dtype}.json"
        print(f"  Downloading {filename} …", end=" ", flush=True)
        data = fetch_json(f"{DOWNLOAD_URL}/{filename}")
        if data is None:
            print("SKIP")
            continue
        # Some country files contain a single object instead of a list; normalise.
        records = data if isinstance(data, list) else [data]
        for r in records:
            if isinstance(r, dict):
                r["_country"] = cc.upper()
        merged.extend(records)
        print(f"OK ({len(records)} records)")
        time.sleep(0.05)   
    return merged


def save(data, dataset_name):
    """Write a merged list to resources/navdata/{dataset_name}.json.gz."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}.json.gz")
    encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    with gzip.open(out_path, "wb") as f:
        f.write(encoded)
    original_kb  = len(encoded) / 1024
    compressed_kb = os.path.getsize(out_path) / 1024
    ratio = compressed_kb / original_kb * 100
    print(f"  ✓ Saved {len(data)} records → {out_path}")
    print(f"    {original_kb:.0f} KB → {compressed_kb:.0f} KB ({ratio:.0f}% of original)\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "types",
        nargs="*",
        metavar="DATASET",
        help="Dataset(s) to download (see list above). Omit to download all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve requested types to short codes
    if args.types:
        requested_codes = set()
        for name in args.types:
            code = NAME_TO_CODE.get(name.lower())
            if code is None:
                print(f"Unknown type '{name}'. Available: {', '.join(sorted(NAME_TO_CODE))}")
                return
            requested_codes.add(code)
    else:
        requested_codes = None   # None means "all"

    # List GCS bucket
    print("Listing objects in the OpenAIP bucket …")
    objects = list_bucket_objects()
    print(f"  Found {len(objects)} objects.\n")

    type_to_countries = find_country_files(objects)

    # Filter to requested types (if any)
    if requested_codes:
        type_to_countries = {k: v for k, v in type_to_countries.items() if k in requested_codes}

    if not type_to_countries:
        print("No matching data types found in the bucket.")
        return

    # Download and save each type
    for dtype in sorted(type_to_countries):
        countries    = type_to_countries[dtype]
        dataset_name = DATA_TYPES.get(dtype, dtype)
        print(f"━━ {dataset_name} ({dtype}) — {len(countries)} countries ━━")
        merged = download_and_merge(dtype, countries)
        save(merged, dataset_name)

    print("Done! All data saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
