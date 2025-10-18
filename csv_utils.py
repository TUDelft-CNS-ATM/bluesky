# csv_utils.py
"""
Utility functions for initializing and writing simulation logs to CSV.
"""

import csv
import os
from datetime import datetime


def init_csv(log_dir: str) -> str:
    """
    Creates the log directory (if missing) and initializes a timestamped CSV file.
    Returns the full CSV path.
    """
    os.makedirs(log_dir, exist_ok=True)
    filename = f"bluesky_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(log_dir, filename)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "simt", "utc", "id", "lat", "lon", "alt", "gs", "hdg"])

    return csv_path


def append_rows(csv_path: str, step: int, data: list[dict]):
    """
    Appends multiple aircraft state rows to the existing CSV.
    Each item in 'data' must be a dictionary with keys:
    id, lat, lon, alt, gs, hdg, simt, utc
    """
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for ac in data:
            writer.writerow([
                step,
                ac.get("simt", 0.0),
                ac.get("utc", ""),
                ac.get("id", ""),
                ac.get("lat", 0.0),
                ac.get("lon", 0.0),
                ac.get("alt", 0.0),
                ac.get("gs", 0.0),
                ac.get("hdg", 0.0)
            ])
