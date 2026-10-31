"""
Walks through every completed game in chronological order, and for each one:
  1. Records the PRE-GAME Elo win probability and both teams' PRE-GAME momentum
     (i.e. only using information that would have been available before kickoff)
  2. THEN updates both trackers with the actual result

This ordering is the whole ballgame for avoiding leakage: if we updated the trackers
first and then recorded their values, we'd be handing the model information from the
very game it's trying to predict.

Produces one row per game with:
  season, week, home_team, away_team, elo_prob_home, home_momentum, away_momentum,
  momentum_diff, actual_home_win
"""

import pandas as pd
from data import get_clean_games
from elo import EloRatingSystem
from momentum import MomentumTracker


def build_feature_table(games: pd.DataFrame, momentum_window: int = 5) -> pd.DataFrame:
    completed = games[games["is_completed"]].copy()

    elo = EloRatingSystem()
    momentum = MomentumTracker(window=momentum_window)

    rows = []
    for _, g in completed.iterrows():
        home, away = g["home_team"], g["away_team"]

        # --- record pre-game state (no leakage) ---
        home_momentum = momentum.get_momentum(home)
        away_momentum = momentum.get_momentum(away)

        # elo.update() itself returns the PRE-game win prob, then updates ratings --
        # so calling it here does both steps in the right order for us.
        elo_prob_home = elo.update(
            season=g["season"], home_team=home, away_team=away,
            home_score=g["home_score"], away_score=g["away_score"],
        )

        # --- now update momentum with the result that just happened ---
        momentum.update(home, away, g["home_score"], g["away_score"])

        rows.append({
            "season": g["season"],
            "week": g["week"],
            "home_team": home,
            "away_team": away,
            "elo_prob_home": elo_prob_home,
            "home_momentum": home_momentum,
            "away_momentum": away_momentum,
            "momentum_diff": home_momentum - away_momentum,
            "actual_home_win": 1 if g["home_score"] > g["away_score"] else 0,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    games = get_clean_games(refresh=False)
    feature_table = build_feature_table(games)
    print(feature_table.shape)
    print(feature_table.tail(10))
    feature_table.to_csv("data/feature_table.csv", index=False)
    print("\nSaved to data/feature_table.csv")
