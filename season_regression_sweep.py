"""
Sweep SEASON_REGRESSION (fraction each team's rating reverts toward 1500 between seasons).
0.0 = no regression, ratings carry over untouched. 1.0 = full reset to 1500 every season.
"""

from sweep_utils import load_sweep_games, sweep_parameter, report_best

if __name__ == "__main__":
    games = load_sweep_games()
    values = [0.0, 0.10, 0.20, 0.25, 1/3, 0.40, 0.50, 0.65, 1.0]
    df = sweep_parameter(games, "season_regression", values)
    report_best(df, "season_regression")
