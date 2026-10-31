"""
Same season-based holdout methodology as train_model.py (train only on seasons strictly
before the test season, evaluate across many holdout seasons and average), but now testing
whether EPA and turnover margin features add value on top of Elo -- the features Elo is
structurally blind to, since Elo only ever sees final scores.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

from build_epa_features import build_epa_feature_table

FEATURE_SETS = {
    "elo_only": ["elo_prob_home"],
    "elo_plus_epa": ["elo_prob_home", "off_epa_diff", "def_epa_diff"],
    "elo_plus_turnovers": ["elo_prob_home", "turnover_margin_diff"],
    "elo_plus_all": ["elo_prob_home", "off_epa_diff", "def_epa_diff", "turnover_margin_diff"],
}
TARGET_COL = "actual_home_win"


def evaluate_holdout_season(feature_table: pd.DataFrame, test_season: int, feature_cols: list) -> dict:
    train = feature_table[feature_table["season"] < test_season]
    test = feature_table[feature_table["season"] == test_season]
    if len(test) == 0 or len(train) == 0:
        return None

    model = LogisticRegression()
    model.fit(train[feature_cols], train[TARGET_COL])

    probs = model.predict_proba(test[feature_cols])[:, 1]
    preds = (probs > 0.5).astype(int)

    return {
        "test_season": test_season,
        "n_games": len(test),
        "accuracy": accuracy_score(test[TARGET_COL], preds),
        "brier": brier_score_loss(test[TARGET_COL], probs),
    }


def run_feature_set(feature_table: pd.DataFrame, feature_cols: list, test_seasons: list) -> pd.DataFrame:
    results = [evaluate_holdout_season(feature_table, s, feature_cols) for s in test_seasons]
    return pd.DataFrame([r for r in results if r is not None])


if __name__ == "__main__":
    feature_table = build_epa_feature_table()

    test_seasons = sorted(feature_table[feature_table["season"] >= 2010]["season"].unique())
    # Only use FULL seasons as test sets (a full regular season + playoffs is ~285 games).
    # A season still in progress would be a small, noisy sample and would distort the averages.
    MIN_GAMES_FOR_TEST_SEASON = 250
    test_seasons = [s for s in test_seasons if
                    (feature_table["season"] == s).sum() >= MIN_GAMES_FOR_TEST_SEASON]

    summary_rows = []
    for name, cols in FEATURE_SETS.items():
        results = run_feature_set(feature_table, cols, test_seasons)
        summary_rows.append({
            "feature_set": name,
            "avg_accuracy": results["accuracy"].mean(),
            "avg_brier": results["brier"].mean(),
        })

    summary = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))
