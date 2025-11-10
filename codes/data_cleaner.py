# This code is ued to clean the Kaggle datasets for Twitch streaming datasets and Steam active players dataset
# Below is how exactly this code works to help our team conduct the data cleaning"
# 1) Load each CSV (robust to encodings)
# 2) Auto-detect a year or date column per file and report which one is used
# 3) Keep only rows from 2019–2021
# 4) Remove fully empty rows
# 5) Remove any row that has missing values
# Output: same file names, saved into "cleaned" subfolder

import pandas as pd
from pathlib import Path

# ===== Config =====
INPUT_FILES = [
    Path("Twitch_game_data.csv"),
    Path("Twitch_global_data.csv"),
    Path("steam_charts.csv"),
]
CLEAN_DIR = Path("cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

YEAR_RANGE = {2019, 2020, 2021}

def read_csv_robust(p: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(p, encoding=enc, sep=None, engine="python", on_bad_lines="skip")
        except Exception:
            continue
    return pd.read_csv(p, encoding="utf-8", sep=None, engine="python",
                       on_bad_lines="skip", encoding_errors="replace")

def detect_year_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    cols = list(df.columns)

    # 1) Exact "year"
    for col in cols:
        if col.lower() == "year":
            y = pd.to_numeric(df[col], errors="coerce")
            return y, col

    # 2) Common datetime headers
    for col in cols:
        if col.lower() in ("date", "timestamp", "datetime"):
            dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if dt.notna().any():
                return dt.dt.year, col

    # 3) Steam Charts style "Month" column
    for col in cols:
        if "month" in col.lower():
            dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if dt.notna().any():
                return dt.dt.year, col

    # 4) Any header containing "year" (e.g., "release_year")
    for col in cols:
        if "year" in col.lower():
            y = pd.to_numeric(df[col], errors="coerce")
            if y.notna().any():
                return y, col

    # 5) Heuristic: try parsing every column as dates and pick the best
    best = None
    best_col = None
    for col in cols:
        dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        valid = dt.notna().sum()
        if valid > 0 and (best is None or valid > best.notna().sum()):
            best = dt
            best_col = col
    if best is not None:
        return best.dt.year, best_col

    # 6) Heuristic: numeric column with many values between 1900 and 2100
    for col in cols:
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().any():
            mask = (y >= 1900) & (y <= 2100)
            if mask.sum() >= len(df) * 0.5:
                return y, col

    raise ValueError("No usable year or date column found")

def clean_df(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    years, src_col = detect_year_series(df)
    df = df[years.isin(YEAR_RANGE)].copy()
    df = df.dropna(how="all")
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="any")
    return df, src_col

def clean_file(in_path: Path, out_dir: Path) -> None:
    df_in = read_csv_robust(in_path)
    df_out, col_used = clean_df(df_in)
    out_path = out_dir / in_path.name  # keep original name
    df_out.to_csv(out_path, index=False)
    print(f"{in_path.name}: used '{col_used}' for year detection | input {len(df_in)} -> cleaned {len(df_out)} -> {out_path}")

for csv_path in INPUT_FILES:
    clean_file(csv_path, CLEAN_DIR)
