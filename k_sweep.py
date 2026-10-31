"""
Test several K_FACTOR values against the full history backtest, so we can pick
the value that actually performs best instead of guessing.
"""

from sweep_utils import load_sweep_games, sweep_parameter, report_best

if __name__ == "__main__":
    games = load_sweep_games()
    values = [5, 10, 15, 20, 25, 30, 40, 60]
    df = sweep_parameter(games, "k_factor", values)
    report_best(df, "k_factor")
