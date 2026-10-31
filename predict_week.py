"""
Run this each week to get predictions for the next set of NFL games.

What it does:
1. Downloads the latest game results + schedule from nflverse.
2. Replays every completed game through the Elo system, in order, to build up
   current team ratings (this naturally "learns" from every result so far).
3. Finds the next set of games that haven't been played yet, and prints a
   win-probability prediction for each, along with the current Vegas line for comparison.
4. Prints the full current power rankings.

Usage:
    python3 predict_week.py
    python3 predict_week.py --season 2026 --week 3   # to see a specific week
"""

import argparse
import pandas as pd
from data import get_clean_games
from elo import EloRatingSystem


def build_ratings_through_present(games: pd.DataFrame) -> EloRatingSystem:
    elo = EloRatingSystem()
    completed = games[games["is_completed"]]
    for _, g in completed.iterrows():
        elo.update(
            season=g["season"],
            home_team=g["home_team"],
            away_team=g["away_team"],
            home_score=g["home_score"],
            away_score=g["away_score"],
        )
    return elo


def find_target_week(games: pd.DataFrame, season: int = None, week: int = None):
    """Default: the earliest not-yet-completed week on the schedule."""
    upcoming = games[~games["is_completed"]].copy()
    if season is not None:
        upcoming = upcoming[upcoming["season"] == season]
    if week is not None:
        upcoming = upcoming[upcoming["week"] == week]
        return upcoming

    if upcoming.empty:
        return upcoming

    next_season = upcoming["season"].min()
    next_week = upcoming[upcoming["season"] == next_season]["week"].min()
    return upcoming[(upcoming["season"] == next_season) & (upcoming["week"] == next_week)]


def format_prediction(elo: EloRatingSystem, row) -> str:
    home, away = row["home_team"], row["away_team"]
    prob_home = elo.predict(home, away)
    favorite, prob = (home, prob_home) if prob_home >= 0.5 else (away, 1 - prob_home)

    line = f"{away} @ {home}  ->  {favorite} favored, {prob:.1%} win probability"

    spread = row.get("spread_line")
    if pd.notna(spread):
        # positive spread_line = home favored by that many points in this dataset
        if spread > 0:
            line += f"   (Vegas: {home} -{spread:g})"
        elif spread < 0:
            line += f"   (Vegas: {away} -{-spread:g})"
        else:
            line += "   (Vegas: pick'em)"
    return line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--rankings", action="store_true", help="also print full power rankings")
    args = parser.parse_args()

    print("Fetching latest data...")
    games = get_clean_games(refresh=True)

    print("Building Elo ratings from full game history...")
    elo = build_ratings_through_present(games)

    target = find_target_week(games, season=args.season, week=args.week)

    if target.empty:
        print("No upcoming games found for that season/week.")
    else:
        season = target["season"].iloc[0]
        week = target["week"].iloc[0]
        print(f"\n=== Predictions: {season} Week {int(week)} ===\n")
        for _, row in target.sort_values("gameday").iterrows():
            print(format_prediction(elo, row))

    if args.rankings:
        print("\n=== Current Power Rankings (Elo) ===\n")
        for i, (team, rating) in enumerate(elo.ratings_table(), start=1):
            print(f"{i:2d}. {team:4s}  {rating:.1f}")


if __name__ == "__main__":
    main()
