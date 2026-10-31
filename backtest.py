"""
Backtest the Elo system: replay history season by season, and measure how well its
pre-game win probabilities matched what actually happened.

Two metrics we care about:
- Accuracy: did we correctly predict the winner? (treating >50% as a pick)
- Brier score: how well-calibrated were our probabilities? (lower is better; 0 = perfect,
  0.25 = equivalent to always guessing 50/50). This matters more than accuracy alone,
  because a model that says "51%" and wins isn't meaningfully better than one that says "90%"
  and loses -- Brier score rewards honest confidence, not just picking correctly.

We also compare against the Vegas closing spread's implied pick, since that's the real
benchmark for "is this model any good."
"""

import numpy as np
from data import get_clean_games, keep_full_seasons
from elo import EloRatingSystem


def run_backtest(start_season=1999, end_season=None):
    games = get_clean_games(refresh=True)
    # Evaluate on fully-played seasons only, so results are reproducible and not distorted
    # by a season that's still in progress (live predictions use every game -- see predict_week.py).
    games = keep_full_seasons(games)
    completed = games[games["is_completed"]].copy()
    if end_season is not None:
        completed = completed[completed["season"] <= end_season]
    completed = completed[completed["season"] >= start_season]

    elo = EloRatingSystem()

    predictions = []
    for _, g in completed.iterrows():
        prob_home_win = elo.update(
            season=g["season"],
            home_team=g["home_team"],
            away_team=g["away_team"],
            home_score=g["home_score"],
            away_score=g["away_score"],
        )
        actual_home_win = 1 if g["home_score"] > g["away_score"] else 0
        predictions.append({
            "season": g["season"],
            "week": g["week"],
            "prob_home_win": prob_home_win,
            "actual_home_win": actual_home_win,
            "spread_line": g.get("spread_line", np.nan),
        })

    import pandas as pd
    pred_df = pd.DataFrame(predictions)

    # Skip each team's first ~32 games league-wide (early weeks of 1999) where ratings
    # are still close to default and not yet meaningful -- optional, but keeps things honest.
    pred_df["elo_correct"] = (pred_df["prob_home_win"] > 0.5).astype(int) == pred_df["actual_home_win"]
    pred_df["brier"] = (pred_df["prob_home_win"] - pred_df["actual_home_win"]) ** 2

    print(f"Games backtested: {len(pred_df)}")
    print(f"Elo accuracy:     {pred_df['elo_correct'].mean():.4f}")
    print(f"Elo Brier score:  {pred_df['brier'].mean():.4f}  (0.25 = coin flip baseline)")

    # Vegas comparison: spread_line is home team's spread (negative = home favored)
    vegas = pred_df.dropna(subset=["spread_line"]).copy()
    # In this dataset, positive spread_line means the HOME team is favored by that many
    # points (opposite of the usual sportsbook convention where favorites are negative).
    vegas["vegas_favors_home"] = (vegas["spread_line"] > 0).astype(int)
    vegas["vegas_correct"] = vegas["vegas_favors_home"] == vegas["actual_home_win"]
    print(f"\nGames with Vegas lines available: {len(vegas)}")
    print(f"Vegas favorite accuracy:          {vegas['vegas_correct'].mean():.4f}")

    return pred_df


if __name__ == "__main__":
    run_backtest()
