"""
Fetches NFL game results (1999-present) from the nflverse public dataset on GitHub.
This includes final scores AND Vegas closing lines, which we'll use later to benchmark
our model against the market.
"""

import os
import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "games_raw.csv")

# Franchises that relocated during the dataset's window (1999-present) get a new
# abbreviation in the raw data. Without this mapping, the Elo system would treat e.g.
# the pre-2016 St. Louis Rams and the current LA Rams as two unrelated teams and lose
# all rating history at the relocation boundary.
TEAM_ALIASES = {
    "STL": "LA",    # Rams: St. Louis -> Los Angeles, 2016
    "SD": "LAC",    # Chargers: San Diego -> Los Angeles, 2017
    "OAK": "LV",    # Raiders: Oakland -> Las Vegas, 2020
}


def load_games(refresh: bool = True) -> pd.DataFrame:
    """
    Load the full games table. By default, always re-downloads the latest copy so that
    this week's newly-completed games and the newly-released upcoming schedule are included
    (this is what makes the script safe to "rerun weekly"). Falls back to the local cache
    if the network fetch fails.
    """
    df = None
    if refresh:
        try:
            df = pd.read_csv(GAMES_URL)
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            df.to_csv(CACHE_PATH, index=False)
        except Exception as e:
            print(f"Could not refresh data from GitHub ({e}); falling back to local cache.")

    if df is None:
        if os.path.exists(CACHE_PATH):
            df = pd.read_csv(CACHE_PATH)
        else:
            # No cache yet (e.g. this is the first script run after a fresh clone) --
            # fetch from source instead of crashing.
            print("No local cache found; fetching game data for the first time...")
            df = pd.read_csv(GAMES_URL)
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            df.to_csv(CACHE_PATH, index=False)

    return df


def clean_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleanup:
    - keep only regular season + playoff games (drop preseason, if any sneaks in)
    - sort chronologically so the Elo system processes games in the right order
    - add a boolean 'is_completed' flag
    """
    df = df.copy()
    df = df[df["game_type"].notna()]
    df["is_completed"] = df["home_score"].notna() & df["away_score"].notna()

    # Normalize relocated franchises to their current abbreviation so Elo ratings
    # carry through continuously across the move.
    df["home_team"] = df["home_team"].replace(TEAM_ALIASES)
    df["away_team"] = df["away_team"].replace(TEAM_ALIASES)

    # gameday sorts games within a week correctly; season/week sorts weeks correctly
    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    return df


def get_clean_games(refresh: bool = True) -> pd.DataFrame:
    return clean_games(load_games(refresh=refresh))


def keep_full_seasons(df: pd.DataFrame, min_games: int = 250) -> pd.DataFrame:
    """
    Keep only games from seasons that are fully played (a complete regular season plus
    playoffs is ~285 games). Used for backtests and sweeps so their results stay
    reproducible and aren't distorted by a season that's still in progress.

    NOT used by predict_week.py -- live ratings should include every game played so far,
    including this season's.
    """
    completed_per_season = df[df["is_completed"]].groupby("season").size()
    full_seasons = completed_per_season[completed_per_season >= min_games].index
    return df[df["season"].isin(full_seasons)]
