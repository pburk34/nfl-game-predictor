"""
Elo rating system for NFL teams.

Core idea: every team has a rating (starts at 1500). Before each game, the rating gap
between the two teams gives us a predicted win probability. After the game, we compare
what actually happened to what we predicted, and nudge both ratings accordingly:
- if a team wins when it "shouldn't have" (huge underdog), its rating jumps a lot
- if a favorite wins as expected, its rating barely moves
- blowouts move ratings more than close games (margin-of-victory multiplier)

This is the same family of system used for chess rankings, and is a surprisingly strong
baseline for NFL prediction because it only needs final scores, not detailed stats.
"""

import math

# Tunable constants. These values are close to what fivethirtyeight used for their
# NFL Elo model, and are a reasonable starting point.
BASE_RATING = 1500.0       # every team starts here (adjusted for returning teams below)
K_FACTOR = 20.0            # how much one game can move a rating; higher = more reactive
HOME_ADVANTAGE = 48.0      # Elo points added to the home team's rating for win-prob purposes
SEASON_REGRESSION = 1.0/3  # how much each team's rating reverts toward the mean between seasons


def expected_win_prob(rating_a: float, rating_b: float) -> float:
    """
    Given two Elo ratings, return team A's probability of winning.
    Standard logistic Elo formula (same as chess).
    """
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def margin_of_victory_multiplier(point_diff: float, elo_diff: float) -> float:
    """
    538's NFL MOV multiplier: blowouts move ratings more, but with diminishing
    returns, and the multiplier shrinks if the favorite was already expected to win big
    (so beating up on a bad team you were already favored against doesn't inflate your rating).
    """
    point_diff = abs(point_diff)
    return math.log(point_diff + 1.0) * (2.2 / ((elo_diff * 0.001) + 2.2))


class EloRatingSystem:
    def __init__(self, k_factor=K_FACTOR, home_advantage=HOME_ADVANTAGE,
                 base_rating=BASE_RATING, season_regression=SEASON_REGRESSION):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.base_rating = base_rating
        self.season_regression = season_regression
        self.ratings = {}          # team -> current Elo rating
        self._current_season = None

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.base_rating)

    def _maybe_regress_for_new_season(self, season):
        """Pull all ratings partway back toward 1500 at the start of a new season,
        since rosters/coaches change and last year's form matters less over time."""
        if self._current_season is not None and season != self._current_season:
            for team in list(self.ratings.keys()):
                r = self.ratings[team]
                self.ratings[team] = r + (self.base_rating - r) * self.season_regression
        self._current_season = season

    def predict(self, home_team: str, away_team: str) -> float:
        """Return home team's win probability for a hypothetical matchup, without updating anything."""
        home_rating = self.get_rating(home_team) + self.home_advantage
        away_rating = self.get_rating(away_team)
        return expected_win_prob(home_rating, away_rating)

    def update(self, season: int, home_team: str, away_team: str,
               home_score: int, away_score: int):
        """
        Process one completed game: predict it, then update both teams' ratings
        based on the actual outcome. Returns the pre-game home win probability
        (useful for backtesting / logging).
        """
        self._maybe_regress_for_new_season(season)

        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        pre_game_home_prob = expected_win_prob(home_rating + self.home_advantage, away_rating)

        if home_score > away_score:
            actual_home_score = 1.0
        elif home_score < away_score:
            actual_home_score = 0.0
        else:
            actual_home_score = 0.5  # ties are rare in the NFL but do happen

        elo_diff = (home_rating + self.home_advantage) - away_rating
        mov_mult = margin_of_victory_multiplier(home_score - away_score, elo_diff)

        shift = self.k_factor * mov_mult * (actual_home_score - pre_game_home_prob)

        self.ratings[home_team] = home_rating + shift
        self.ratings[away_team] = away_rating - shift

        return pre_game_home_prob

    def ratings_table(self):
        """Return current ratings sorted best to worst, as a list of (team, rating) tuples."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
