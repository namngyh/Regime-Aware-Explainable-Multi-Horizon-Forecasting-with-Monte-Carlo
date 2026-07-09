import csv
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_price(tokens, pos):
    token = tokens[pos].strip()
    if pos + 1 < len(tokens) and token.isdigit() and len(token) <= 2:
        next_token = tokens[pos + 1].strip()
        if next_token.replace(".", "", 1).isdigit():
            if "." in next_token:
                integer, decimal = next_token.split(".", 1)
                return float(f"{token}{integer.zfill(3)}.{decimal}"), pos + 2
            return float(f"{token}{next_token.zfill(3)}"), pos + 2
    if "." in token:
        return float(token), pos + 1
    return float(token), pos + 1


def _parse_volume(parts):
    clean = [p.strip() for p in parts if p.strip()]
    if not clean:
        return np.nan
    first, *rest = clean
    return int(first + "".join(part.zfill(3) for part in rest))


def load_vnindex_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for raw in reader:
            if not raw or not raw[0].strip():
                continue
            tokens = [part.strip() for part in raw[1:] if part.strip()]
            if len(tokens) < 5:
                continue
            pos = 0
            open_, pos = _parse_price(tokens, pos)
            high, pos = _parse_price(tokens, pos)
            low, pos = _parse_price(tokens, pos)
            close, pos = _parse_price(tokens, pos)
            volume = _parse_volume(tokens[pos:])
            rows.append(
                {
                    "date": pd.to_datetime(raw[0].strip(), dayfirst=True),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
    df = df.reset_index(drop=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df
