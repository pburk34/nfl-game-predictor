"""
Downloads play-by-play data for all seasons and caches it locally as a parquet file.
This is a much bigger pull than the game-level results in data.py (1.2M+ individual plays
vs ~7,500 games), so we cache it to disk rather than re-fetching on every run the way
predict_week.py refetches game results.

By default this pulls 1999 through the current calendar year, so rerunning it during the
season picks up newly played games. If the current year's data isn't published yet (e.g.
early in the offseason), it automatically falls back to the previous year.
"""

import datetime
import os
import nfl_data_py as nfl

PBP_PATH = os.path.join(os.path.dirname(__file__), "data", "pbp_raw.parquet")

# Only pulling the columns we actually need keeps this fast and small -- the full
# play-by-play schema has 300+ columns including detailed player participation data
# we don't need for EPA/turnover aggregation.
COLUMNS = ["game_id", "season", "week", "posteam", "defteam", "epa",
           "interception", "fumble_lost", "play_type", "season_type"]


def _pull(start_year: int, end_year: int):
    years = list(range(start_year, end_year + 1))
    print(f"Fetching play-by-play data for {start_year}-{end_year}...")
    return nfl.import_pbp_data(
        years, columns=COLUMNS, downcast=True, cache=False, include_participation=False
    )


def fetch_pbp(start_year: int = 1999, end_year: int = None) -> None:
    if end_year is None:
        end_year = datetime.date.today().year

    try:
        df = _pull(start_year, end_year)
    except Exception as e:
        # Current season may not be published yet -- retry through the prior season.
        print(f"Could not fetch through {end_year} ({type(e).__name__}); "
              f"retrying through {end_year - 1}...")
        end_year -= 1
        df = _pull(start_year, end_year)

    os.makedirs(os.path.dirname(PBP_PATH), exist_ok=True)
    df.to_parquet(PBP_PATH)
    print(f"Saved {len(df):,} plays (through {end_year}) to {PBP_PATH}")


if __name__ == "__main__":
    fetch_pbp()
