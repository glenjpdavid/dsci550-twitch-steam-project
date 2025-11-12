
# This code helps my group loads merged_steam_twitch.csv, detects metric columns for Steam Avg Players and Twitch Avg Viewers,
# builds a monthly Date column, and generates PNG visualizations with consistent colors:
# - Twitch is blue, Steam is red.
# Outputs:
# 1) Overall platform trends 2019 to 2021 (monthly totals for Twitch vs Steam, blue vs red)
# 2) Line charts for 5 to 10 notable games (Twitch figure in blue; Steam figure in red)
# 3) Combined per game charts with Twitch and Steam on the same timeline (blue vs red)
# 4) Scatterplots showing correlation between viewers and players (axes labeled in blue vs red)
# 5) Seasonality by month averaged across 2019 to 2021 (blue vs red)
# 6) Optional genre colored scatter if a Genre column exists

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

CSV_PATH = "merged_steam_twitch.csv"
OUT_DIR  = "figures"
SELECTED_GAMES = []

TWITCH_BASE_COLOR = "blue"
STEAM_BASE_COLOR  = "red"

def detect_metric_columns(df):
    cols = [c.strip() for c in df.columns]
    steam_candidates = [c for c in cols if (("avg" in c.lower() and "player" in c.lower()) or ("players" in c.lower() and "avg" in c.lower())) and not c.lower().endswith("_twitch")]
    steam_col = next((c for c in steam_candidates if c.lower().endswith("_steam")), None) or (steam_candidates[0] if steam_candidates else None)
    twitch_candidates = [c for c in cols if (("avg" in c.lower() and "viewer" in c.lower()) or ("viewers" in c.lower() and "avg" in c.lower())) and not c.lower().endswith("_steam")]
    twitch_col = next((c for c in twitch_candidates if c.lower().endswith("_twitch")), None) or (twitch_candidates[0] if twitch_candidates else None)
    if twitch_col is None:
        alt = [c for c in cols if "hours_watched" in c.lower()]
        twitch_col = alt[0] if alt else None
    if steam_col is None or twitch_col is None:
        raise ValueError(f"Could not infer metric columns. Found: {cols}")
    return steam_col, twitch_col

def thousands(x, _pos):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def tab20_colors(n, offset):
    cmap = plt.cm.get_cmap("tab20")
    idx = list(range(offset, 20, 2))
    if n <= len(idx):
        use = idx[:n]
    else:
        reps = int(np.ceil(n / len(idx)))
        use = (idx * reps)[:n]
    return [cmap(i) for i in use]

def color_mapping_for_games(games, platform):
    if platform == "twitch":
        cols = tab20_colors(len(games), offset=0)
    else:
        cols = tab20_colors(len(games), offset=1)
    return {g: cols[i] for i, g in enumerate(games)}

def main():
    ensure_outdir(OUT_DIR)
    df = pd.read_csv(CSV_PATH)
    need = {"Year", "Month", "Game"}
    if not need.issubset(df.columns):
        raise ValueError(f"Merged file must contain {need}. Found: {df.columns.tolist()}")
    steam_col, twitch_col = detect_metric_columns(df)
    work = df.copy()
    work["Year"]  = pd.to_numeric(work["Year"], errors="coerce").astype("Int64")
    work["Month"] = pd.to_numeric(work["Month"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["Year", "Month", "Game"])
    work["Date"] = pd.to_datetime(dict(year=work["Year"].astype(int), month=work["Month"].astype(int), day=1))
    work[steam_col]  = pd.to_numeric(work[steam_col], errors="coerce")
    work[twitch_col] = pd.to_numeric(work[twitch_col], errors="coerce")
    if "hours_watched" in twitch_col.lower():
        days = work["Date"].dt.days_in_month
        work["Twitch_Avg_Viewers_Derived"] = work[twitch_col] / (days * 24.0)
        twitch_plot_col = "Twitch_Avg_Viewers_Derived"
    else:
        twitch_plot_col = twitch_col
    steam_plot_col = steam_col
    work = work[(work["Date"] >= "2019-01-01") & (work["Date"] <= "2021-12-31")].copy()
    if SELECTED_GAMES:
        chosen = [g for g in SELECTED_GAMES if g in work["Game"].unique()]
        if not chosen:
            raise ValueError("None of the SELECTED_GAMES found in the dataset. Leave list empty to auto select.")
        top_games = chosen
    else:
        game_scores = work.groupby("Game").agg({steam_plot_col: "mean", twitch_plot_col: "mean"}).fillna(0.0)
        for c in [steam_plot_col, twitch_plot_col]:
            m = game_scores[c].max()
            game_scores[c] = (game_scores[c] / m) if (m and np.isfinite(m) and m > 0) else 0.0
        game_scores["score"] = 0.5 * game_scores[steam_plot_col] + 0.5 * game_scores[twitch_plot_col]
        top_games = game_scores.sort_values("score", ascending=False).head(8).index.tolist()
    twitch_colors = color_mapping_for_games(top_games, "twitch")
    steam_colors  = color_mapping_for_games(top_games, "steam")
    fmt = FuncFormatter(thousands)

    monthly_totals = work.groupby("Date", as_index=False).agg({twitch_plot_col: "sum", steam_plot_col: "sum"}).rename(columns={twitch_plot_col: "Total_Twitch_Avg_Viewers", steam_plot_col: "Total_Steam_Avg_Players"})
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    ax.plot(monthly_totals["Date"], monthly_totals["Total_Twitch_Avg_Viewers"], label="Twitch total", color=TWITCH_BASE_COLOR, linewidth=2)
    ax.plot(monthly_totals["Date"], monthly_totals["Total_Steam_Avg_Players"], label="Steam total", color=STEAM_BASE_COLOR, linewidth=2)
    per_game = work[work["Game"].isin(top_games)].groupby(["Date", "Game"], as_index=False).agg({twitch_plot_col: "sum", steam_plot_col: "sum"})
    for g in top_games:
        gd = per_game[per_game["Game"] == g]
        ax.plot(gd["Date"], gd[twitch_plot_col], color=twitch_colors[g], linewidth=1.4, label=f"{g} (Twitch)")
    for g in top_games:
        gd = per_game[per_game["Game"] == g]
        ax.plot(gd["Date"], gd[steam_plot_col], color=steam_colors[g], linewidth=1.4, linestyle="--", label=f"{g} (Steam)")
    ax.set_title("2019 to 2021 Overall Trends Totals plus Top Games")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly total")
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "overall_trends_totals_plus_top_games.png"), dpi=160)
    plt.close()

    sel = work[work["Game"].isin(top_games)].copy()
    pivot_t = sel.pivot_table(index="Date", columns="Game", values=twitch_plot_col, aggfunc="mean")
    pivot_s = sel.pivot_table(index="Date", columns="Game", values=steam_plot_col,   aggfunc="mean")

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for g in pivot_t.columns:
        ax.plot(pivot_t.index, pivot_t[g], label=g, color=twitch_colors[g], linewidth=1.8)
    ax.set_title("Top Games Twitch Average Viewers Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average viewers")
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(loc="best", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "games_twitch_avg_viewers.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for g in pivot_s.columns:
        ax.plot(pivot_s.index, pivot_s[g], label=g, color=steam_colors[g], linewidth=1.8)
    ax.set_title("Top Games Steam Average Players Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average players")
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(loc="best", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "games_steam_avg_players.png"), dpi=160)
    plt.close()

    combined_dir = os.path.join(OUT_DIR, "per_game_combined")
    ensure_outdir(combined_dir)
    for g in top_games:
        gdf = work[work["Game"] == g].copy()
        if gdf.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax1.plot(gdf["Date"], gdf[twitch_plot_col], label="Twitch Avg Viewers", color=TWITCH_BASE_COLOR, linewidth=1.8)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Twitch Avg Viewers", color=TWITCH_BASE_COLOR)
        ax1.tick_params(axis='y', labelcolor=TWITCH_BASE_COLOR)
        ax1.yaxis.set_major_formatter(fmt)
        ax2 = ax1.twinx()
        ax2.plot(gdf["Date"], gdf[steam_plot_col], label="Steam Avg Players", color=STEAM_BASE_COLOR, linewidth=1.8)
        ax2.set_ylabel("Steam Avg Players", color=STEAM_BASE_COLOR)
        ax2.tick_params(axis='y', labelcolor=STEAM_BASE_COLOR)
        ax2.yaxis.set_major_formatter(fmt)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
        plt.title(f"{g} Twitch and Steam Over Time")
        plt.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_\-]+","_", g)
        plt.savefig(os.path.join(combined_dir, f"{safe}_twitch_and_steam.png"), dpi=160)
        plt.close()

    scatter_df = work.dropna(subset=[twitch_plot_col, steam_plot_col]).copy()
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ax.scatter(scatter_df[twitch_plot_col], scatter_df[steam_plot_col], s=12, alpha=0.5)
    ax.set_title("Correlation Twitch Viewers vs Steam Players 2019 to 2021")
    ax.set_xlabel("Twitch Avg Viewers", color=TWITCH_BASE_COLOR)
    ax.set_ylabel("Steam Avg Players", color=STEAM_BASE_COLOR)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scatter_views_vs_players_overall.png"), dpi=160)
    plt.close()

    for yr in [2019, 2020, 2021]:
        sub = scatter_df[scatter_df["Year"] == yr]
        if len(sub) == 0:
            continue
        plt.figure(figsize=(9, 7))
        ax = plt.gca()
        ax.scatter(sub[twitch_plot_col], sub[steam_plot_col], s=12, alpha=0.5)
        ax.set_title(f"Correlation Twitch Viewers vs Steam Players {yr}")
        ax.set_xlabel("Twitch Avg Viewers", color=TWITCH_BASE_COLOR)
        ax.set_ylabel("Steam Avg Players", color=STEAM_BASE_COLOR)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"scatter_views_vs_players_{yr}.png"), dpi=160)
        plt.close()

    season = work.copy()
    season["MonthName"] = season["Date"].dt.month_name()
    season_totals = season.groupby(["Month", "MonthName"], as_index=False).agg({twitch_plot_col: "sum", steam_plot_col: "sum"}).sort_values("Month")
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    ax.plot(season_totals["MonthName"], season_totals[twitch_plot_col], marker="o", label="Twitch total", color=TWITCH_BASE_COLOR, linewidth=1.8)
    ax.plot(season_totals["MonthName"], season_totals[steam_plot_col], marker="o", label="Steam total", color=STEAM_BASE_COLOR, linewidth=1.8)
    ax.set_title("Seasonality by Month Twitch vs Steam (Totals 2019 to 2021)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total")
    ax.yaxis.set_major_formatter(fmt)
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "seasonality_monthly_totals.png"), dpi=160)
    plt.close()

    if "Genre" in work.columns:
        genre_scatter_dir = os.path.join(OUT_DIR, "genre_scatter")
        ensure_outdir(genre_scatter_dir)
        gdf = work.dropna(subset=[twitch_plot_col, steam_plot_col, "Genre"]).copy()
        if not gdf.empty:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            for genre, sub in gdf.groupby("Genre"):
                ax.scatter(sub[twitch_plot_col], sub[steam_plot_col], s=12, alpha=0.5, label=str(genre))
            ax.set_title("Correlation by Genre Twitch Viewers vs Steam Players")
            ax.set_xlabel("Twitch Avg Viewers", color=TWITCH_BASE_COLOR)
            ax.set_ylabel("Steam Avg Players", color=STEAM_BASE_COLOR)
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            ax.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(genre_scatter_dir, "scatter_by_genre.png"), dpi=160)
            plt.close()

    print("Detected Steam metric:", steam_col)
    print("Detected Twitch metric:", twitch_col)
    print("Selected games:", top_games)
    print(f"Figures saved to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
