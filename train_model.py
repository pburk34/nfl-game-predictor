"""
Trains a logistic regression on [elo_prob_home, momentum_diff] -> home_win, and compares
it against pure Elo on the exact same held-out games.

Train/test split: by SEASON, not randomly. We train on all seasons before some cutoff and
test only on seasons after it. This matters a lot here -- a random row-by-row split would
let the model train on, say, Week 10 of the 2020 season and test on Week 3 of that same
season, which leaks information across a season the model shouldn't have "lived through"
yet at test time. A season-based split is the honest way to simulate "how would this model
have performed on a season it hadn't seen."

We test across several holdout seasons (not just one) and average results, since any
single season's result is noisy on its own.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

from data import get_clean_games
from build_features import build_feature_table

FEATURE_COLS = ["elo_prob_home", "momentum_diff"]
TARGET_COL = "actual_home_win"


def evaluate_holdout_season(feature_table: pd.DataFrame, test_season: int) -> dict:
    train = feature_table[feature_table["season"] < test_season]
    test = feature_table[feature_table["season"] == test_season]

    if len(test) == 0 or len(train) == 0:
        return None

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    model_probs = model.predict_proba(X_test)[:, 1]
    model_preds = (model_probs > 0.5).astype(int)

    # Pure-Elo baseline on the SAME test games, for a fair apples-to-apples comparison
    elo_probs = test["elo_prob_home"].values
    elo_preds = (elo_probs > 0.5).astype(int)

    return {
        "test_season": test_season,
        "n_games": len(test),
        "model_accuracy": accuracy_score(y_test, model_preds),
        "model_brier": brier_score_loss(y_test, model_probs),
        "elo_accuracy": accuracy_score(y_test, elo_preds),
        "elo_brier": brier_score_loss(y_test, elo_probs),
        "momentum_coef": model.coef_[0][1],   # how much weight momentum got
        "elo_coef": model.coef_[0][0],
    }


if __name__ == "__main__":
    games = get_clean_games(refresh=False)
    feature_table = build_feature_table(games)

    # Test on every season from 2010 onward, training only on seasons strictly before it.
    # (Skipping earlier seasons as test sets since there's little training history behind them.)
    test_seasons = sorted(feature_table[feature_table["season"] >= 2010]["season"].unique())
    # Only use FULL seasons as test sets (a full regular season + playoffs is ~285 games).
    # A season still in progress would be a small, noisy sample and would distort the averages.
    MIN_GAMES_FOR_TEST_SEASON = 250
    test_seasons = [s for s in test_seasons if
                    (feature_table["season"] == s).sum() >= MIN_GAMES_FOR_TEST_SEASON]

    results = [evaluate_holdout_season(feature_table, s) for s in test_seasons]
    results = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)

    pd.set_option("display.width", 120)
    print(results_df.to_string(index=False))

    print("\n=== Averages across all holdout seasons ===")
    print(f"Model accuracy: {results_df['model_accuracy'].mean():.4f}   "
          f"Elo accuracy: {results_df['elo_accuracy'].mean():.4f}")
    print(f"Model Brier:    {results_df['model_brier'].mean():.4f}   "
          f"Elo Brier:    {results_df['elo_brier'].mean():.4f}")
    print(f"\nAvg momentum coefficient: {results_df['momentum_coef'].mean():.4f}")
    print(f"Avg elo coefficient:      {results_df['elo_coef'].mean():.4f}")
