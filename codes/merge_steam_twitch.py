# This code helps our group loads cleaned steam_charts.csv and Twitch_game_data.csv, parses Steam "Month" text into numeric Year and Month,
# normalizes Twitch "Year" and "Month", then INNER JOINs on ["Game","Year","Month"].
# Columns are ordered as: Month, Year, Game, then all Steam columns (non-keys), then all Twitch columns (non-keys).
# Overlapping names get _steam and _twitch suffixes. Rows without a match in both files are excluded.
# Output is sorted by Year, Month, Game and written to merged_steam_twitch.csv.

import re
import pandas as pd
from calendar import month_abbr, month_name

STEAM_PATH = "steam_charts.csv"
TWITCH_PATH = "Twitch_game_data.csv"
OUT_PATH   = "merged_steam_twitch.csv"

def month_to_int(val):
    """Convert month strings or numerics to 1-12; return pd.NA if not parseable."""
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if s.isdigit():
        m = int(s)
        return m if 1 <= m <= 12 else pd.NA
    s_low = s.lower()
    for i in range(1, 13):
        if s_low == month_name[i].lower() or s_low == month_abbr[i].lower():
            return i
    return pd.NA

def extract_year_month_from_text(text_val):
    """Extract (year, month) from strings like 'December 2019' or '2019-12'."""
    if pd.isna(text_val):
        return (pd.NA, pd.NA)
    s = str(text_val).strip()
    # Try pandas datetime first
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return (int(dt.year), int(dt.month))
    # Fallback regex for "Month Year"
    m = re.match(r"([A-Za-z]+)\s+(\d{4})$", s)
    if m:
        m_str, y_str = m.groups()
        m_int = month_to_int(m_str)
        return (int(y_str), m_int if pd.notna(m_int) else pd.NA)
    # Fallback regex for "YYYY-MM" or "YYYY/M/D"
    m2 = re.match(r"(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?$", s)
    if m2:
        y_int = int(m2.group(1))
        m_int = int(m2.group(2))
        return (y_int, m_int if 1 <= m_int <= 12 else pd.NA)
    return (pd.NA, pd.NA)

def order_columns(merged, steam_keyed, twitch_keyed):
    keys = ["Month", "Year", "Game"]
    steam_nonkey = [c for c in steam_keyed.columns if c not in keys]
    twitch_nonkey = [c for c in twitch_keyed.columns if c not in keys]
    overlaps = set(steam_nonkey).intersection(twitch_nonkey)

    def map_name(col, source):
        if col in overlaps:
            return f"{col}_steam" if source == "steam" else f"{col}_twitch"
        return col

    ordered = keys + [map_name(c, "steam") for c in steam_nonkey] + [map_name(c, "twitch") for c in twitch_nonkey]

    # Deduplicate and retain only columns that exist
    seen, final_cols = set(), []
    for c in ordered:
        if c in merged.columns and c not in seen:
            final_cols.append(c)
            seen.add(c)
    # Append any leftover columns defensively
    for c in merged.columns:
        if c not in seen:
            final_cols.append(c)
            seen.add(c)
    return final_cols

def main():
    steam = pd.read_csv(STEAM_PATH)
    twitch = pd.read_csv(TWITCH_PATH)

    # Preserve original Steam Month and build Year + Month ints
    steam["_Month_original_steam"] = steam.get("Month")
    ys, ms = [], []
    for val in steam["_Month_original_steam"]:
        y, m = extract_year_month_from_text(val)
        ys.append(y)
        ms.append(m)
    steam["Year"] = ys
    steam["Month"] = ms

    # Normalize key fields
    if "Game" in steam.columns:
        steam["Game"] = steam["Game"].astype(str).str.strip()
    if "Game" in twitch.columns:
        twitch["Game"] = twitch["Game"].astype(str).str.strip()

    if "Year" in twitch.columns:
        twitch["Year"] = pd.to_numeric(twitch["Year"], errors="coerce").astype("Int64")
    else:
        twitch["Year"] = pd.NA

    if "Month" in twitch.columns:
        twitch["Month"] = twitch["Month"].apply(month_to_int).astype("Int64")
    else:
        twitch["Month"] = pd.NA

    # Drop rows missing keys so unmatched entries are removed
    steam_keyed = steam.dropna(subset=["Game", "Year", "Month"]).copy()
    twitch_keyed = twitch.dropna(subset=["Game", "Year", "Month"]).copy()

    # Ensure consistent dtypes for merge keys
    steam_keyed["Year"] = steam_keyed["Year"].astype("int64")
    steam_keyed["Month"] = steam_keyed["Month"].astype("int64")
    twitch_keyed["Year"] = twitch_keyed["Year"].astype("int64")
    twitch_keyed["Month"] = twitch_keyed["Month"].astype("int64")

    # Inner join to keep only matched rows
    merged = pd.merge(
        steam_keyed,
        twitch_keyed,
        on=["Game", "Year", "Month"],
        how="inner",
        suffixes=("_steam", "_twitch")
    )

    # Sort and reorder columns
    merged_sorted = merged.sort_values(by=["Year", "Month", "Game"]).reset_index(drop=True)
    final_cols = order_columns(merged_sorted, steam_keyed, twitch_keyed)
    merged_sorted = merged_sorted[final_cols]

    merged_sorted.to_csv(OUT_PATH, index=False)
    print(f"Merged rows: {len(merged_sorted):,}")
    print(f"Saved to: {OUT_PATH}")
    print("First 10 columns:", final_cols[:10])

if __name__ == "__main__":
    main()
