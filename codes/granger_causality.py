# Granger Causality between Steam Peak Players and Twitch Peak Viewers
# Dataset: merged_steam_twitch.csv

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# ---------------------------
# 1) Config
# ---------------------------
FILE_PATH = "merged_steam_twitch.csv"

# For monthly data, 1-3 lags is a common starting point
MAX_LAG = 3

# Minimum observations per game to run a reasonable test
# You can raise this if you want stricter filtering
MIN_OBS = 12

# If True, use first differences to reduce non-stationarity risk
# Granger tests often work better on stationary series
USE_DIFFERENCE = False

# ---------------------------
# 2) Load data
# ---------------------------
df = pd.read_csv(FILE_PATH)

# Keep only what we need for this analysis
cols_needed = ["Game", "Year", "Month", "Peak Players.Steam", "Peak_viewers.Twitch"]
df = df[cols_needed].copy()

# ---------------------------
# 3) Handle duplicated Game-Year-Month rows
# ---------------------------
# If your merge created duplicates, take the max of the peak metrics
df = df.groupby(["Game", "Year", "Month"], as_index=False).max()

# Build a monthly date column for proper sorting
df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))

# Sort for time series order within each game
df = df.sort_values(["Game", "date"])

# ---------------------------
# 4) Helper function
# ---------------------------
def run_granger_for_game(game_df, max_lag=3, use_diff=False):
    """
    Runs Granger causality tests in both directions for one game.

    statsmodels rule:
    grangercausalitytests(data, maxlag)
    tests whether column 2 Granger-causes column 1.

    We will test:
    A) Twitch -> Steam  (data = [steam, twitch])
    B) Steam  -> Twitch (data = [twitch, steam])

    Returns:
    dict with p-values by lag for each direction
    """
    # Drop missing rows
    g = game_df.dropna(subset=["Peak Players.Steam", "Peak_viewers.Twitch"]).copy()

    # Convert to float
    g["steam"] = g["Peak Players.Steam"].astype(float)
    g["twitch"] = g["Peak_viewers.Twitch"].astype(float)

    # Optionally difference the series
    if use_diff:
        g["steam"] = g["steam"].diff()
        g["twitch"] = g["twitch"].diff()
        g = g.dropna()

    # Need enough observations to run the chosen lag length
    if len(g) <= max_lag + 2:
        return None

    # A) Twitch -> Steam
    # column 2 causes column 1
    data_ts = g[["steam", "twitch"]]
    res_ts = grangercausalitytests(data_ts, maxlag=max_lag, verbose=False)
    pvals_ts = {lag: res_ts[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)}

    # B) Steam -> Twitch
    data_st = g[["twitch", "steam"]]
    res_st = grangercausalitytests(data_st, maxlag=max_lag, verbose=False)
    pvals_st = {lag: res_st[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)}

    return {
        "pvals_twitch_causes_steam": pvals_ts,
        "pvals_steam_causes_twitch": pvals_st,
        "n_obs_used": len(g)
    }

# ---------------------------
# 5) Run per-game tests
# ---------------------------
summary_rows = []

for game, gdf in df.groupby("Game"):
    # Make sure we have enough raw observations
    gdf_non_missing = gdf.dropna(subset=["Peak Players.Steam", "Peak_viewers.Twitch"])
    if len(gdf_non_missing) < MIN_OBS:
        continue

    result = run_granger_for_game(gdf, max_lag=MAX_LAG, use_diff=USE_DIFFERENCE)
    if result is None:
        continue

    p_ts = result["pvals_twitch_causes_steam"]
    p_st = result["pvals_steam_causes_twitch"]

    # Store a compact summary: best (lowest) p-value and corresponding lag
    summary_rows.append({
        "Game": game,
        "n_obs_used": result["n_obs_used"],
        "min_p_twitch_causes_steam": min(p_ts.values()),
        "best_lag_twitch_causes_steam": min(p_ts, key=p_ts.get),
        "min_p_steam_causes_twitch": min(p_st.values()),
        "best_lag_steam_causes_twitch": min(p_st, key=p_st.get),
    })

summary = pd.DataFrame(summary_rows)

# Sort to see strongest evidence first (Steam -> Twitch)
summary = summary.sort_values("min_p_steam_causes_twitch", ascending=True)

# ---------------------------
# 6) Output
# ---------------------------
print("Granger causality summary (per game):")
print(summary.head(30))

# Save for your report appendix
summary.to_csv("granger_summary_per_game.csv", index=False)
print("\nSaved: granger_summary_per_game.csv")

# ---------------------------
# 7) Optional: inspect one game in detail
# ---------------------------
# Uncomment and change the game name to view all lag p-values
"""
TARGET_GAME = "Counter-Strike: Global Offensive"
one = df[df["Game"] == TARGET_GAME].sort_values("date")
detail = run_granger_for_game(one, max_lag=MAX_LAG, use_diff=USE_DIFFERENCE)

print(f"\nDetailed results for {TARGET_GAME}:")
print("Twitch -> Steam p-values by lag:", detail["pvals_twitch_causes_steam"])
print("Steam -> Twitch p-values by lag:", detail["pvals_steam_causes_twitch"])
"""
