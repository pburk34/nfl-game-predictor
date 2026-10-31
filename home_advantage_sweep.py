"""Sweep HOME_ADVANTAGE (Elo points added to the home team before computing win prob)."""

from sweep_utils import load_sweep_games, sweep_parameter, report_best

if __name__ == "__main__":
    games = load_sweep_games()
    values = [0, 15, 25, 35, 48, 55, 65, 80, 100]
    df = sweep_parameter(games, "home_advantage", values)
    report_best(df, "home_advantage")
