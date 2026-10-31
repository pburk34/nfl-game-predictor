"""
Aggregates raw play-by-play data (data/pbp_raw.parquet, built by fetch_pbp.py) into one
row per team per game, with:
  - off_epa_per_play: average EPA on plays this team ran on offense (pass/run only --
    excludes special teams, kneels, and spikes, which aren't meaningful measures of a
    team's offensive quality)
  - def_epa_per_play: average EPA this team ALLOWED on defense (lower is better on defense --
    we'll flip the sign in feature building so "higher = better" is consistent across
    features)
  - turnovers_lost: interceptions thrown + fumbles lost while on offense (giveaways)
  - turnovers_forced: interceptions + fumbles recovered while on defense (takeaways)
  - turnover_margin: turnovers_forced - turnovers_lost, for that single game

This produces GAME-LEVEL stats (what actually happened in a game the team already played).
build_epa_features.py then turns these into PRE-GAME rolling averages for prediction --
using this game's own stats to predict this game would be leakage.
"""

import os
import pandas as pd

PBP_PATH = "data/pbp_raw.parquet"

# Only count normal scrimmage plays for EPA -- excludes kickoffs, punts, extra points,
# field goals, kneels, and spikes, none of which reflect offensive/defensive quality
# the way a snap-by-snap pass or run does.
SCRIMMAGE_PLAY_TYPES = ["pass", "run"]


def build_team_game_stats(pbp_path: str = PBP_PATH) -> pd.DataFrame:
    if not os.path.exists(pbp_path):
        raise FileNotFoundError(
            f"No play-by-play data found at '{pbp_path}'. Run `python fetch_pbp.py` "
            f"first to download and cache it (only needs to be done once, or whenever "
            f"you want to refresh with newly played games)."
        )
    pbp = pd.read_parquet(pbp_path)
    scrimmage = pbp[pbp["play_type"].isin(SCRIMMAGE_PLAY_TYPES)].copy()
    scrimmage = scrimmage.dropna(subset=["posteam", "defteam", "epa"])

    # --- offensive side: group by the team that had the ball ---
    off = scrimmage.groupby(["game_id", "posteam"]).agg(
        off_epa_per_play=("epa", "mean"),
        off_plays=("epa", "size"),
        turnovers_lost=("interception", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})

    fumbles_lost = scrimmage.groupby(["game_id", "posteam"])["fumble_lost"].sum().reset_index()
    fumbles_lost = fumbles_lost.rename(columns={"posteam": "team", "fumble_lost": "fumbles_lost"})
    off = off.merge(fumbles_lost, on=["game_id", "team"], how="left")
    off["turnovers_lost"] = off["turnovers_lost"] + off["fumbles_lost"].fillna(0)
    off = off.drop(columns=["fumbles_lost"])

    # --- defensive side: group by the team that was defending ---
    deff = scrimmage.groupby(["game_id", "defteam"]).agg(
        def_epa_per_play=("epa", "mean"),
        def_plays=("epa", "size"),
        turnovers_forced=("interception", "sum"),
    ).reset_index().rename(columns={"defteam": "team"})

    fumbles_forced = scrimmage.groupby(["game_id", "defteam"])["fumble_lost"].sum().reset_index()
    fumbles_forced = fumbles_forced.rename(columns={"defteam": "team", "fumble_lost": "fumbles_forced"})
    deff = deff.merge(fumbles_forced, on=["game_id", "team"], how="left")
    deff["turnovers_forced"] = deff["turnovers_forced"] + deff["fumbles_forced"].fillna(0)
    deff = deff.drop(columns=["fumbles_forced"])

    # --- combine into one row per team per game ---
    team_game = off.merge(deff, on=["game_id", "team"], how="outer")
    team_game["turnover_margin"] = team_game["turnovers_forced"] - team_game["turnovers_lost"]

    return team_game


if __name__ == "__main__":
    stats = build_team_game_stats()
    print(stats.shape)
    print(stats.head(10))
    stats.to_csv("data/team_game_stats.csv", index=False)
    print("\nSaved to data/team_game_stats.csv")
