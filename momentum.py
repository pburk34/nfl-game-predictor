"""
Tracks each team's recent scoring momentum: the average point differential (points scored
minus points allowed) over their last N games. The idea: a team that's been winning big
lately might be genuinely better than their Elo rating reflects yet (Elo moves gradually
by design), or a team on a losing skid might be worse than their rating shows.

Like elo.py, this is designed to be updated one game at a time, in chronological order,
so that the "momentum" value used to predict any given game only ever reflects games that
happened strictly before it -- no lookahead / leakage.
"""

from collections import deque

DEFAULT_WINDOW = 5  # how many recent games to average over


class MomentumTracker:
    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window
        self.history = {}  # team -> deque of recent point differentials (most recent last)

    def get_momentum(self, team: str) -> float:
        """Average point differential over the team's last `window` games.
        Returns 0.0 (neutral) if the team has no prior games yet."""
        recent = self.history.get(team)
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def update(self, home_team: str, away_team: str, home_score: int, away_score: int):
        """Call this AFTER using get_momentum() for both teams to make a prediction --
        this records the just-finished game's result into each team's rolling window."""
        home_diff = home_score - away_score
        away_diff = away_score - home_score

        for team, diff in [(home_team, home_diff), (away_team, away_diff)]:
            if team not in self.history:
                self.history[team] = deque(maxlen=self.window)
            self.history[team].append(diff)
