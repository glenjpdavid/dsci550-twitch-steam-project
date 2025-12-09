# VAR + IRF pipeline with Granger filter
#
# This single script generates all submission files without relying on
# intermediate scripts.
#
# Inputs:
# - granger_summary_per_game.csv
# - merged_steam_twitch.csv
#
# Outputs:
# 1) granger_filtered_for_var.csv
# 2) var_irf_quick_summary.csv
# 3) var_model_summaries.txt
# 4) irf_plots/  (PNG files)
# 5) granger_var_combined_summary.csv  (final one-file table)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# ---------------------------
# 1) Config
# ---------------------------
GRANGER_SUMMARY_PATH = "granger_summary_per_game.csv"
MERGED_TS_PATH = "merged_steam_twitch.csv"

P_THRESHOLD = 0.05
MIN_OBS = 12
MAX_LAG_ORDER = 6
USE_DIFFERENCE = False
IRF_HORIZON = 6

# Optional cap for a smaller appendix
# Set to None to keep all Granger-significant games
TOP_N_GAMES = None

# Ranking rule if TOP_N_GAMES is used
# Options: "steam_to_twitch" or "twitch_to_steam"
TOP_RANK_DIRECTION = "steam_to_twitch"

# Output names
OUT_GRANGER_FILTERED = "granger_filtered_for_var.csv"
OUT_IRF_SUMMARY = "var_irf_quick_summary.csv"
OUT_VAR_TEXT = "var_model_summaries.txt"
OUT_COMBINED = "granger_var_combined_summary.csv"
IRF_PLOT_DIR = "irf_plots"

# ---------------------------
# 2) Prepare output folder
# ---------------------------
os.makedirs(IRF_PLOT_DIR, exist_ok=True)

# ---------------------------
# 3) Load Granger summary and select games
# ---------------------------
summary = pd.read_csv(GRANGER_SUMMARY_PATH)

sig = summary[
    (summary["min_p_twitch_causes_steam"] < P_THRESHOLD) |
    (summary["min_p_steam_causes_twitch"] < P_THRESHOLD)
].copy()

# Rank for optional TOP_N selection
if TOP_RANK_DIRECTION == "twitch_to_steam":
    sig = sig.sort_values("min_p_twitch_causes_steam", ascending=True)
else:
    sig = sig.sort_values("min_p_steam_causes_twitch", ascending=True)

if TOP_N_GAMES is not None:
    sig = sig.head(TOP_N_GAMES)

# Save filtered Granger table
sig.to_csv(OUT_GRANGER_FILTERED, index=False)
selected_games = sig["Game"].tolist()

print(f"Saved: {OUT_GRANGER_FILTERED}")
print(f"Selected games for VAR attempt: {len(selected_games)}")

# ---------------------------
# 4) Load merged time series data
# ---------------------------
df = pd.read_csv(MERGED_TS_PATH)

df = df[["Game", "Year", "Month", "Peak Players.Steam", "Peak_viewers.Twitch"]].copy()

# Resolve duplicates per Game-Year-Month by taking max peaks
df = df.groupby(["Game", "Year", "Month"], as_index=False).max()

# Monthly time index
df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))

# ---------------------------
# 5) Helper: prepare per-game series
# ---------------------------
def prepare_game_series(game_name):
    """
    Creates a two-variable monthly time series for one game:
    - steam: Peak Players on Steam
    - twitch: Peak Viewers on Twitch
    """
    g = df[df["Game"] == game_name].copy()
    g = g.sort_values("date")

    g = g[["date", "Peak Players.Steam", "Peak_viewers.Twitch"]].dropna()
    g = g.rename(columns={
        "Peak Players.Steam": "steam",
        "Peak_viewers.Twitch": "twitch"
    })

    g = g.set_index("date")

    # Optional differencing for robustness checks
    if USE_DIFFERENCE:
        g = g.diff().dropna()

    return g

# ---------------------------
# 6) Fit VAR and save outputs
# ---------------------------
results_rows = []
modeled_games = 0

with open(OUT_VAR_TEXT, "w", encoding="utf-8") as f:
    f.write("VAR model summaries\n")
    f.write(f"Granger filter rule: p < {P_THRESHOLD} in either direction\n")
    f.write("Data frequency: monthly\n")
    f.write(f"IRF horizon (months): {IRF_HORIZON}\n\n")

    for game in selected_games:
        g = prepare_game_series(game)

        if len(g) < MIN_OBS:
            continue

        model = VAR(g)

        # Select lag order by AIC with a small cap
        try:
            order_res = model.select_order(MAX_LAG_ORDER)
            selected_lag = int(order_res.aic)
            if selected_lag < 1:
                selected_lag = 1
        except Exception:
            selected_lag = 1

        # Fit model with fallback
        try:
            var_res = model.fit(selected_lag)
        except Exception:
            try:
                selected_lag = 1
                var_res = model.fit(selected_lag)
            except Exception:
                continue

        modeled_games += 1

        # Write summary to text file
        f.write("====================================================\n")
        f.write(f"Game: {game}\n")
        f.write(f"Observations used: {len(g)}\n")
        f.write(f"Selected VAR lag (AIC-based, capped at {MAX_LAG_ORDER}): {selected_lag}\n\n")
        f.write(str(var_res.summary()))
        f.write("\n\n")

        # IRF for Twitch shock -> Steam response
        irf = var_res.irf(IRF_HORIZON)

        # Save IRF plot
        try:
            irf.plot(orth=False, impulse="twitch", response="steam")
            plt.title(f"IRF (monthly): Twitch shock -> Steam response\n{game}")
            plt.tight_layout()

            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in game).strip()
            plot_path = os.path.join(IRF_PLOT_DIR, f"{safe_name}_IRF_twitch_to_steam.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
        except Exception:
            plt.close()

        # Save numeric IRF snapshot
        try:
            irf_array = irf.irfs  # (horizon+1, response, impulse)
            steam_idx = list(g.columns).index("steam")
            twitch_idx = list(g.columns).index("twitch")

            response_series = irf_array[:, steam_idx, twitch_idx]

            row = {
                "Game": game,
                "n_obs_var": len(g),
                "selected_lag": selected_lag,
            }

            # Save irf_0..irf_6 (or up to horizon)
            for h in range(0, IRF_HORIZON + 1):
                key = f"irf_{h}"
                row[key] = response_series[h] if h < len(response_series) else np.nan

            results_rows.append(row)
        except Exception:
            pass

# ---------------------------
# 7) Save VAR IRF summary CSV
# ---------------------------
if results_rows:
    var_irf = pd.DataFrame(results_rows)
    var_irf = var_irf.sort_values(["n_obs_var", "Game"], ascending=[False, True])
    var_irf.to_csv(OUT_IRF_SUMMARY, index=False)
    print(f"Saved: {OUT_IRF_SUMMARY}")
else:
    var_irf = pd.DataFrame(columns=["Game", "n_obs_var", "selected_lag"])
    var_irf.to_csv(OUT_IRF_SUMMARY, index=False)
    print(f"Saved empty IRF summary: {OUT_IRF_SUMMARY}")

print(f"Saved: {OUT_VAR_TEXT}")
print(f"Saved IRF plots folder: {IRF_PLOT_DIR}/")
print(f"Games successfully modeled with VAR: {modeled_games}")

# ---------------------------
# 8) Merge into one final combined CSV
# ---------------------------
granger_filtered = pd.read_csv(OUT_GRANGER_FILTERED)
var_irf = pd.read_csv(OUT_IRF_SUMMARY)

combined = granger_filtered.merge(var_irf, on="Game", how="left")

preferred_order = [
    "Game",
    "n_obs_used",
    "min_p_twitch_causes_steam",
    "best_lag_twitch_causes_steam",
    "min_p_steam_causes_twitch",
    "best_lag_steam_causes_twitch",
    "n_obs_var",
    "selected_lag",
]

# Add IRF columns in order if present
for h in range(0, IRF_HORIZON + 1):
    preferred_order.append(f"irf_{h}")

existing_preferred = [c for c in preferred_order if c in combined.columns]
remaining = [c for c in combined.columns if c not in existing_preferred]
combined = combined[existing_preferred + remaining]

combined.to_csv(OUT_COMBINED, index=False)

print(f"Saved final combined CSV: {OUT_COMBINED}")
