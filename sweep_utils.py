"""
Generalized sweep: test a range of values for any single EloRatingSystem hyperparameter
against the full backtest, holding everything else at default. Reused by the individual
sweep scripts (k_sweep.py, home_advantage_sweep.py, season_regression_sweep.py).

Sweeps run on fully-played seasons only (see data.keep_full_seasons), so results are
reproducible and not distorted by a season that's still in progress.

Note: this is a one-at-a-time sweep (only one parameter changes per run). It won't catch
interaction effects between parameters (e.g. maybe the best home_advantage is different
once K_FACTOR is also changed) -- that would require a full grid search. One-at-a-time is
a reasonable first pass and much cheaper to run and reason about.
"""

import pandas as pd
from data import get_clean_games, keep_full_seasons
from elo import EloRatingSystem


def load_sweep_games() -> pd.DataFrame:
    """Cached game data, restricted to fully-played seasons."""
    return keep_full_seasons(get_clean_games(refresh=False))


def backtest_with_params(games: pd.DataFrame, **elo_kwargs) -> dict:
    completed = games[games["is_completed"]]
    elo = EloRatingSystem(**elo_kwargs)

    correct = 0
    brier_sum = 0.0
    n = 0

    for _, g in completed.iterrows():
        prob_home_win = elo.update(
            season=g["season"],
            home_team=g["home_team"],
            away_team=g["away_team"],
            home_score=g["home_score"],
            away_score=g["away_score"],
        )
        actual_home_win = 1 if g["home_score"] > g["away_score"] else 0
        predicted_home_win = 1 if prob_home_win > 0.5 else 0

        correct += int(predicted_home_win == actual_home_win)
        brier_sum += (prob_home_win - actual_home_win) ** 2
        n += 1

    return {"accuracy": correct / n, "brier_score": brier_sum / n}


def sweep_parameter(games: pd.DataFrame, param_name: str, values: list) -> pd.DataFrame:
    rows = []
    for v in values:
        result = backtest_with_params(games, **{param_name: v})
        result[param_name] = v
        rows.append(result)
    df = pd.DataFrame(rows)[[param_name, "accuracy", "brier_score"]]
    return df


def report_best(df: pd.DataFrame, param_name: str):
    print(df.to_string(index=False))
    best_acc = df.loc[df["accuracy"].idxmax()]
    best_brier = df.loc[df["brier_score"].idxmin()]
    print(f"\nBest accuracy:    {param_name}={best_acc[param_name]}  ({best_acc['accuracy']:.4f})")
    print(f"Best Brier score: {param_name}={best_brier[param_name]}  ({best_brier['brier_score']:.4f})")
