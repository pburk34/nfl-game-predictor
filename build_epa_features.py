"""
Turns per-game team stats (from aggregate_pbp.py) into PRE-GAME rolling averages, the same
way momentum.py turns game results into pre-game momentum. For each team, we track a
rolling window of their last N games' off_epa_per_play, def_epa_per_play, and
turnover_margin, and use the average of the PAST games only -- never the game being
predicted -- as the feature value.

Combines these with Elo into one feature table for training, same structure as
build_features.py but with EPA/turnover columns added.
"""

from collections import deque
import pandas as pd
from data import get_clean_games
from elo import EloRatingSystem
from aggregate_pbp import build_team_game_stats

WINDOW = 5


class RollingStatsTracker:
    """Generic version of momentum.py's tracker -- tracks a rolling average for
    however many named stats you give it, per team."""

    def __init__(self, stat_names, window: int = WINDOW):
        self.stat_names = stat_names
        self.window = window
        self.history = {name: {} for name in stat_names}

    def get(self, team: str) -> dict:
        result = {}
        for name in self.stat_names:
            recent = self.history[name].get(team)
            result[name] = (sum(recent) / len(recent)) if recent else 0.0
        return result

    def update(self, team: str, stat_values: dict):
        for name in self.stat_names:
            if team not in self.history[name]:
                self.history[name][team] = deque(maxlen=self.window)
            self.history[name][team].append(stat_values[name])


def build_epa_feature_table(momentum_window: int = WINDOW) -> pd.DataFrame:
    games = get_clean_games(refresh=False)
    completed = games[games["is_completed"]].copy()

    team_game_stats = build_team_game_stats()
    # index by (game_id, team) for fast lookup of "what this team did in this game"
    stats_lookup = team_game_stats.set_index(["game_id", "team"])

    elo = EloRatingSystem()
    tracker = RollingStatsTracker(
        stat_names=["off_epa_per_play", "def_epa_per_play", "turnover_margin"],
        window=momentum_window,
    )

    rows = []
    skipped_no_pbp = 0
    for _, g in completed.iterrows():
        home, away, game_id = g["home_team"], g["away_team"], g["game_id"]

        # --- pre-game feature snapshot (no leakage) ---
        home_pre = tracker.get(home)
        away_pre = tracker.get(away)
        elo_prob_home = elo.predict(home, away)  # snapshot BEFORE this game's Elo update

        rows.append({
            "game_id": game_id,
            "season": g["season"],
            "week": g["week"],
            "home_team": home,
            "away_team": away,
            "elo_prob_home": elo_prob_home,
            "home_off_epa": home_pre["off_epa_per_play"],
            "away_off_epa": away_pre["off_epa_per_play"],
            "home_def_epa": home_pre["def_epa_per_play"],
            "away_def_epa": away_pre["def_epa_per_play"],
            "home_turnover_margin": home_pre["turnover_margin"],
            "away_turnover_margin": away_pre["turnover_margin"],
            "actual_home_win": 1 if g["home_score"] > g["away_score"] else 0,
        })

        # --- now update Elo with the real result ---
        elo.update(season=g["season"], home_team=home, away_team=away,
                   home_score=g["home_score"], away_score=g["away_score"])

        # --- update rolling EPA/turnover trackers, if we have pbp data for this game ---
        for team in (home, away):
            key = (game_id, team)
            if key in stats_lookup.index:
                row = stats_lookup.loc[key]
                tracker.update(team, {
                    "off_epa_per_play": row["off_epa_per_play"],
                    "def_epa_per_play": row["def_epa_per_play"],
                    "turnover_margin": row["turnover_margin"],
                })
            else:
                skipped_no_pbp += 1

    if skipped_no_pbp:
        print(f"Note: {skipped_no_pbp} team-games had no play-by-play stats available "
              f"(rolling window just carries fewer games for those teams).")

    df = pd.DataFrame(rows)
    # Derived, model-friendly versions: higher = better for offense, and flip defense
    # sign so "higher = better defense" too, matching the intuitive direction of turnover margin.
    df["off_epa_diff"] = df["home_off_epa"] - df["away_off_epa"]
    df["def_epa_diff"] = df["away_def_epa"] - df["home_def_epa"]  # away allowing more = good for home
    df["turnover_margin_diff"] = df["home_turnover_margin"] - df["away_turnover_margin"]
    return df


if __name__ == "__main__":
    df = build_epa_feature_table()
    print(df.shape)
    print(df.tail(8)[["season", "week", "home_team", "away_team",
                       "off_epa_diff", "def_epa_diff", "turnover_margin_diff", "actual_home_win"]])
    df.to_csv("data/epa_feature_table.csv", index=False)
    print("\nSaved to data/epa_feature_table.csv")
