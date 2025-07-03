import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TeamStrengthCalculator:

    def __init__(self, min_games=3, time_decay=0.01, form_window=5):
        """
        Enhanced team strength calculator with time decay and improved metrics

        Args:
            min_games: Minimum games required for reliable statistics
            time_decay: Exponential decay factor for time weighting (higher = more recent bias)
            form_window: Number of recent games for form calculation
        """
        self.min_games = min_games
        self.time_decay = time_decay
        self.form_window = form_window

    def calculate_strengths(self, results_df):
        """
        Calculate enhanced team strengths with time weighting and advanced metrics
        """
        try:
            if results_df.empty:
                print("Warning: Empty results dataframe")
                return pd.DataFrame()

            # Ensure date column is datetime
            if 'Date' in results_df.columns:
                results_df = results_df.copy()
                results_df['Date'] = pd.to_datetime(results_df['Date'])
                results_df = results_df.sort_values('Date')

            # Get all teams
            home_teams = set(results_df['HomeTeam'].dropna())
            away_teams = set(results_df['AwayTeam'].dropna())
            all_teams = list(home_teams | away_teams)

            if not all_teams:
                print("Warning: No teams found in data")
                return pd.DataFrame()

            # Initialize team statistics dataframe
            team_stats = pd.DataFrame(index=all_teams)

            # Calculate time weights if date information is available
            if 'Date' in results_df.columns:
                weights = self._calculate_time_weights(results_df)
            else:
                weights = np.ones(len(results_df))
                print("Warning: No date column found, using uniform weights")

            # Calculate basic statistics for each team
            for team in all_teams:
                stats = self._calculate_team_stats(team, results_df, weights)
                for key, value in stats.items():
                    team_stats.loc[team, key] = value

            # Calculate league averages with time weighting
            league_avg_goals = self._calculate_weighted_league_average(
                results_df, weights)
            team_stats['league_avg'] = league_avg_goals

            # Calculate relative strengths
            team_stats = self._calculate_relative_strengths(
                team_stats, league_avg_goals)

            # Calculate advanced metrics
            team_stats = self._calculate_advanced_metrics(
                team_stats, results_df, weights)

            # Add original column names for backward compatibility
            team_stats['total_goals_scored'] = team_stats[
                'avg_goals_scored'] * team_stats['total_games']
            team_stats['total_goals_conceded'] = team_stats[
                'avg_goals_conceded'] * team_stats['total_games']

            # Calculate recent form (keep original method name behavior)
            team_stats['recent_form'] = team_stats.apply(
                lambda x: self._calculate_recent_form(x.name, results_df,
                                                      weights),
                axis=1)

            # Round all numeric columns for readability
            numeric_columns = team_stats.select_dtypes(
                include=[np.number]).columns
            team_stats[numeric_columns] = team_stats[numeric_columns].round(3)

            print(
                f"✅ Enhanced team strengths calculated for {len(all_teams)} teams"
            )
            return team_stats

        except Exception as e:
            print(f"Error calculating enhanced team strengths: {e}")
            return pd.DataFrame()

    def _calculate_time_weights(self, results_df):
        """Calculate exponential time decay weights"""
        try:
            if 'Date' not in results_df.columns:
                return np.ones(len(results_df))

            # Get the most recent date
            max_date = results_df['Date'].max()

            # Calculate days from most recent match
            days_from_recent = (max_date - results_df['Date']).dt.days

            # Apply exponential decay: more recent matches get higher weight
            weights = np.exp(-self.time_decay * days_from_recent)

            # Normalize weights to sum to number of matches (maintains scale)
            weights = weights * len(weights) / weights.sum()

            return weights

        except Exception as e:
            print(f"Error calculating time weights: {e}")
            return np.ones(len(results_df))

    def _calculate_team_stats(self, team, results_df, weights):
        """Calculate weighted statistics for a single team"""
        stats = {}

        # Home matches
        home_matches = results_df[results_df['HomeTeam'] == team].copy()
        home_weights = weights[results_df['HomeTeam'] == team]

        if len(home_matches) > 0:
            stats['home_games'] = len(home_matches)
            stats['home_goals_scored'] = np.average(home_matches['FTHG'],
                                                    weights=home_weights)
            stats['home_goals_conceded'] = np.average(home_matches['FTAG'],
                                                      weights=home_weights)
            stats['home_points'] = np.average(home_matches.apply(
                self._calculate_points_home, axis=1),
                                              weights=home_weights)
            # Keep original column names
            stats['home_goals_avg'] = stats['home_goals_scored']
            stats['home_goals_conceded_avg'] = stats['home_goals_conceded']
        else:
            stats.update({
                'home_games': 0,
                'home_goals_scored': 0,
                'home_goals_avg': 0,
                'home_goals_conceded': 0,
                'home_goals_conceded_avg': 0,
                'home_points': 0
            })

        # Away matches
        away_matches = results_df[results_df['AwayTeam'] == team].copy()
        away_weights = weights[results_df['AwayTeam'] == team]

        if len(away_matches) > 0:
            stats['away_games'] = len(away_matches)
            stats['away_goals_scored'] = np.average(away_matches['FTAG'],
                                                    weights=away_weights)
            stats['away_goals_conceded'] = np.average(away_matches['FTHG'],
                                                      weights=away_weights)
            stats['away_points'] = np.average(away_matches.apply(
                self._calculate_points_away, axis=1),
                                              weights=away_weights)
            # Keep original column names
            stats['away_goals_avg'] = stats['away_goals_scored']
            stats['away_goals_conceded_avg'] = stats['away_goals_conceded']
        else:
            stats.update({
                'away_games': 0,
                'away_goals_scored': 0,
                'away_goals_avg': 0,
                'away_goals_conceded': 0,
                'away_goals_conceded_avg': 0,
                'away_points': 0
            })

        # Combined statistics
        total_games = stats['home_games'] + stats['away_games']
        if total_games > 0:
            # Weight home and away stats by number of games
            home_weight = stats['home_games'] / total_games
            away_weight = stats['away_games'] / total_games

            stats['total_games'] = total_games
            stats['avg_goals_scored'] = (
                stats['home_goals_scored'] * home_weight +
                stats['away_goals_scored'] * away_weight)
            stats['avg_goals_conceded'] = (
                stats['home_goals_conceded'] * home_weight +
                stats['away_goals_conceded'] * away_weight)
            stats['avg_points'] = (stats['home_points'] * home_weight +
                                   stats['away_points'] * away_weight)
            stats['goal_difference'] = stats['avg_goals_scored'] - stats[
                'avg_goals_conceded']
        else:
            stats.update({
                'total_games': 0,
                'avg_goals_scored': 0,
                'avg_goals_conceded': 0,
                'avg_points': 0,
                'goal_difference': 0
            })

        return stats

    def _calculate_points_home(self, row):
        """Calculate points for home team"""
        if row['FTHG'] > row['FTAG']:
            return 3  # Win
        elif row['FTHG'] == row['FTAG']:
            return 1  # Draw
        else:
            return 0  # Loss

    def _calculate_points_away(self, row):
        """Calculate points for away team"""
        if row['FTAG'] > row['FTHG']:
            return 3  # Win
        elif row['FTAG'] == row['FTHG']:
            return 1  # Draw
        else:
            return 0  # Loss

    def _calculate_weighted_league_average(self, results_df, weights):
        """Calculate weighted league average goals per game"""
        try:
            total_goals = results_df['FTHG'] + results_df['FTAG']
            weighted_avg = np.average(total_goals, weights=weights) / 2
            return max(0.5, weighted_avg)  # Minimum sensible value
        except:
            return 1.4  # Fallback value

    def _calculate_relative_strengths(self, team_stats, league_avg_goals):
        """Calculate attack and defense strengths relative to league average"""
        # Attack strength (goals scored relative to league average)
        team_stats['attack_strength'] = np.where(
            league_avg_goals > 0,
            team_stats['avg_goals_scored'] / league_avg_goals, 1.0)

        # Defense strength (goals conceded relative to league average)
        team_stats['defense_strength'] = np.where(
            league_avg_goals > 0,
            team_stats['avg_goals_conceded'] / league_avg_goals, 1.0)

        # Home/Away specific strengths (use overall if insufficient games)
        team_stats['home_attack_strength'] = np.where(
            team_stats['home_games'] >= self.min_games,
            team_stats['home_goals_scored'] / league_avg_goals,
            team_stats['attack_strength'])

        team_stats['away_attack_strength'] = np.where(
            team_stats['away_games'] >= self.min_games,
            team_stats['away_goals_scored'] / league_avg_goals,
            team_stats['attack_strength'])

        team_stats['home_defense_strength'] = np.where(
            team_stats['home_games'] >= self.min_games,
            team_stats['home_goals_conceded'] / league_avg_goals,
            team_stats['defense_strength'])

        team_stats['away_defense_strength'] = np.where(
            team_stats['away_games'] >= self.min_games,
            team_stats['away_goals_conceded'] / league_avg_goals,
            team_stats['defense_strength'])

        # Ensure all strengths are positive
        strength_cols = [
            'attack_strength', 'defense_strength', 'home_attack_strength',
            'away_attack_strength', 'home_defense_strength',
            'away_defense_strength'
        ]

        for col in strength_cols:
            team_stats[col] = np.maximum(0.1, team_stats[col])

        return team_stats

    def _calculate_advanced_metrics(self, team_stats, results_df, weights):
        """Calculate advanced performance metrics"""
        for team in team_stats.index:
            # Performance consistency (coefficient of variation of points)
            team_stats.loc[team, 'consistency'] = self._calculate_consistency(
                team, results_df, weights)

            # Strength of schedule faced
            team_stats.loc[team, 'strength_of_schedule'] = self._calculate_sos(
                team, results_df, team_stats)

        return team_stats

    def _calculate_recent_form(self, team, results_df, weights=None):
        """Calculate weighted recent form (last N games) - keeps original method signature"""
        try:
            # Get all matches for the team
            team_matches = pd.concat([
                results_df[results_df['HomeTeam'] == team].assign(
                    team_goals=results_df['FTHG'],
                    opp_goals=results_df['FTAG'],
                    is_home=True),
                results_df[results_df['AwayTeam'] == team].assign(
                    team_goals=results_df['FTAG'],
                    opp_goals=results_df['FTHG'],
                    is_home=False)
            ])

            if 'Date' in results_df.columns:
                team_matches = team_matches.sort_values('Date')

            if len(team_matches) == 0:
                return 0.0

            # Take last N games
            recent_matches = team_matches.tail(self.form_window)

            # Calculate points from recent matches
            def get_points(row):
                if row['team_goals'] > row['opp_goals']:
                    return 3
                elif row['team_goals'] == row['opp_goals']:
                    return 1
                else:
                    return 0

            recent_points = recent_matches.apply(get_points, axis=1)

            # Return points per game in recent form
            return recent_points.mean() if len(recent_points) > 0 else 0.0

        except Exception as e:
            return 1.0  # Default neutral form

    def _calculate_consistency(self, team, results_df, weights):
        """Calculate performance consistency (lower is more consistent)"""
        try:
            # Get all matches for the team
            team_matches = pd.concat([
                results_df[results_df['HomeTeam'] == team].assign(
                    team_goals=results_df['FTHG'],
                    opp_goals=results_df['FTAG']),
                results_df[results_df['AwayTeam'] == team].assign(
                    team_goals=results_df['FTAG'],
                    opp_goals=results_df['FTHG'])
            ])

            if len(team_matches) < 3:
                return 1.0  # Default value for insufficient data

            # Calculate points for each match
            def get_points(row):
                if row['team_goals'] > row['opp_goals']:
                    return 3
                elif row['team_goals'] == row['opp_goals']:
                    return 1
                else:
                    return 0

            points = team_matches.apply(get_points, axis=1)

            # Calculate coefficient of variation (std/mean)
            if points.mean() > 0:
                return points.std() / points.mean()
            else:
                return 1.0

        except Exception as e:
            return 1.0

    def _calculate_sos(self, team, results_df, team_stats):
        """Calculate strength of schedule (average opponent strength)"""
        try:
            opponents = []

            # Get home opponents
            home_opponents = results_df[results_df['HomeTeam'] ==
                                        team]['AwayTeam']
            opponents.extend(home_opponents.tolist())

            # Get away opponents
            away_opponents = results_df[results_df['AwayTeam'] ==
                                        team]['HomeTeam']
            opponents.extend(away_opponents.tolist())

            if not opponents:
                return 1.0

            # Calculate average attack strength of opponents faced
            opponent_strengths = []
            for opponent in opponents:
                if opponent in team_stats.index:
                    strength = team_stats.loc[opponent, 'attack_strength']
                    opponent_strengths.append(strength)

            return np.mean(opponent_strengths) if opponent_strengths else 1.0

        except Exception as e:
            return 1.0
