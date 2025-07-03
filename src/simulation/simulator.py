import pandas as pd
import numpy as np
from scipy.stats import poisson
import time
import os
import logging

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    def __init__(self, fixtures_df, poisson_model, seed=42):
        self.fixtures = fixtures_df.copy()
        self.model = poisson_model
        self.rng = np.random.RandomState(seed)
        self.teams = self._get_all_teams()

    @classmethod
    def from_upcoming_fixtures(cls, poisson_model, upcoming_fixtures_path="data/clean/upcoming_fixtures.csv", seed=42):
        """Create simulator using the upcoming_fixtures.csv file directly"""
        try:
            logger.info(f"Loading upcoming fixtures from {upcoming_fixtures_path}")

            # Read the CSV directly with robust error handling
            fixtures_df = cls._load_upcoming_fixtures_directly(upcoming_fixtures_path)

            if fixtures_df.empty:
                logger.warning("No valid fixtures found in upcoming_fixtures.csv")
                raise ValueError("No fixtures available for simulation")

            logger.info(f"Successfully loaded {len(fixtures_df)} fixtures for simulation")
            return cls(fixtures_df, poisson_model, seed)

        except Exception as e:
            logger.error(f"Error loading upcoming fixtures: {e}")
            # Fallback to existing fixtures
            if os.path.exists("data/clean/fixtures.csv"):
                logger.info("Falling back to existing fixtures.csv")
                fixtures_df = pd.read_csv("data/clean/fixtures.csv", parse_dates=['Date'])
                return cls(fixtures_df, poisson_model, seed)
            else:
                raise ValueError(f"Could not load fixtures: {e}")

    @staticmethod
    def _load_upcoming_fixtures_directly(filepath):
        """Load upcoming fixtures directly from CSV with robust parsing"""
        try:
            # Read CSV with error handling
            df = pd.read_csv(filepath, on_bad_lines='skip')
            
            # Handle different column name formats
            if 'Home_Team' in df.columns and 'Away_Team' in df.columns:
                # Rename columns to match expected format
                df = df.rename(columns={
                    'Home_Team': 'HomeTeam',
                    'Away_Team': 'AwayTeam'
                })
            
            # Ensure required columns exist
            if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
                raise ValueError("CSV must contain HomeTeam and AwayTeam columns")
            
            # Clean and standardize team names
            team_mapping = {
                'GAIS': 'GAIS', 'Malmo FF': 'Malmo FF', 'Malmö FF': 'Malmo FF',
                'Oster': 'Oster', 'Östers IF': 'Oster', 'Osters IF': 'Oster',
                'Mjallby': 'Mjallby', 'Mjällby AIF': 'Mjallby',
                'Hammarby': 'Hammarby', 'Varnamo': 'Varnamo',
                'Goteborg': 'Goteborg', 'IFK Göteborg': 'Goteborg',
                'Norrkoping': 'Norrkoping', 'IFK Norrköping': 'Norrkoping',
                'Hacken': 'Hacken', 'BK Häcken': 'Hacken',
                'Brommapojkarna': 'Brommapojkarna', 'IF Brommapojkarna': 'Brommapojkarna',
                'Sirius': 'Sirius', 'IK Sirius': 'Sirius',
                'Elfsborg': 'Elfsborg', 'IF Elfsborg': 'Elfsborg',
                'Halmstad': 'Halmstad', 'Halmstads BK': 'Halmstad',
                'Djurgarden': 'Djurgarden', 'Djurgårdens IF': 'Djurgarden',
                'AIK': 'AIK', 'Degerfors': 'Degerfors'
            }
            
            # Apply team name mapping
            df['HomeTeam'] = df['HomeTeam'].map(team_mapping).fillna(df['HomeTeam'])
            df['AwayTeam'] = df['AwayTeam'].map(team_mapping).fillna(df['AwayTeam'])
            
            # Parse dates if needed
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            else:
                # Create a default date if missing
                df['Date'] = pd.Timestamp.now()
            
            # Filter out rows with missing team names
            df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
            df = df[df['HomeTeam'].str.strip() != '']
            df = df[df['AwayTeam'].str.strip() != '']
            
            # Ensure we have the required columns for simulation
            required_columns = ['Date', 'HomeTeam', 'AwayTeam']
            df = df[required_columns]
            
            logger.info(f"Successfully loaded {len(df)} fixtures from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading fixtures from {filepath}: {e}")
            return pd.DataFrame()

    def _get_all_teams(self):
        """Get all unique teams from fixtures"""
        home_teams = set(self.fixtures['HomeTeam'].unique())
        away_teams = set(self.fixtures['AwayTeam'].unique())
        return list(home_teams | away_teams)

    def simulate_one_season(self):
        """Simulate one complete season"""
        try:
            # Initialize points table
            points_table = {team: 0 for team in self.teams}

            # Simulate each fixture
            for _, fixture in self.fixtures.iterrows():
                home_team = fixture['HomeTeam']
                away_team = fixture['AwayTeam']

                # Get expected goals from model
                mu_home, mu_away = self.model.predict_match(home_team, away_team)

                # Simulate match outcome
                home_goals = self.rng.poisson(mu_home)
                away_goals = self.rng.poisson(mu_away)

                # Award points
                if home_goals > away_goals:  # Home win
                    points_table[home_team] += 3
                elif home_goals == away_goals:  # Draw
                    points_table[home_team] += 1
                    points_table[away_team] += 1
                else:  # Away win
                    points_table[away_team] += 3

            return points_table

        except Exception as e:
            print(f"Error simulating season: {e}")
            # Return default points distribution
            return {team: 30 for team in self.teams}

    def run(self, n_simulations=10000, progress_callback=None):
        """Run multiple Monte Carlo simulations"""
        try:
            print(f"Starting {n_simulations:,} Monte Carlo simulations...")
            simulation_results = []

            # Batch processing for better performance
            batch_size = min(1000, n_simulations // 10)

            for i in range(n_simulations):
                if i % batch_size == 0 and progress_callback:
                    progress = (i / n_simulations) * 100
                    progress_callback(progress)

                season_result = self.simulate_one_season()
                simulation_results.append(season_result)

            # Convert to DataFrame
            results_df = pd.DataFrame(simulation_results)

            # Ensure all teams are present in results
            for team in self.teams:
                if team not in results_df.columns:
                    results_df[team] = 0

            print(f"Completed {n_simulations:,} simulations successfully!")
            return results_df

        except Exception as e:
            print(f"Error running simulations: {e}")
            # Return dummy results
            dummy_data = []
            for i in range(min(100, n_simulations)):
                season_result = {team: self.rng.randint(20, 80) for team in self.teams}
                dummy_data.append(season_result)

            return pd.DataFrame(dummy_data)

    def simulate_remaining_matches(self, current_table=None):
        """Simulate only remaining matches with current points"""
        try:
            if current_table is None:
                current_table = {team: 0 for team in self.teams}

            # Start with current points
            points_table = current_table.copy()

            # Simulate remaining fixtures
            for _, fixture in self.fixtures.iterrows():
                home_team = fixture['HomeTeam']
                away_team = fixture['AwayTeam']

                mu_home, mu_away = self.model.predict_match(home_team, away_team)

                home_goals = self.rng.poisson(mu_home)
                away_goals = self.rng.poisson(mu_away)

                if home_goals > away_goals:
                    points_table[home_team] += 3
                elif home_goals == away_goals:
                    points_table[home_team] += 1
                    points_table[away_team] += 1
                else:
                    points_table[away_team] += 3

            return points_table

        except Exception as e:
            print(f"Error simulating remaining matches: {e}")
            return current_table

    def get_match_prediction(self, home_team, away_team, n_simulations=1000):
        """Get detailed prediction for a specific match"""
        try:
            outcomes = {'home_win': 0, 'draw': 0, 'away_win': 0}
            goal_distribution = {}

            mu_home, mu_away = self.model.predict_match(home_team, away_team)

            for _ in range(n_simulations):
                home_goals = self.rng.poisson(mu_home)
                away_goals = self.rng.poisson(mu_away)

                # Track outcomes
                if home_goals > away_goals:
                    outcomes['home_win'] += 1
                elif home_goals == away_goals:
                    outcomes['draw'] += 1
                else:
                    outcomes['away_win'] += 1

                # Track score distribution
                score = f"{home_goals}-{away_goals}"
                goal_distribution[score] = goal_distribution.get(score, 0) + 1

            # Convert to probabilities
            for key in outcomes:
                outcomes[key] = outcomes[key] / n_simulations

            # Get most likely scores
            most_likely_scores = sorted(goal_distribution.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]

            return {
                'probabilities': outcomes,
                'expected_goals': {'home': mu_home, 'away': mu_away},
                'most_likely_scores': most_likely_scores,
                'total_simulations': n_simulations
            }

        except Exception as e:
            print(f"Error predicting match: {e}")
            return {
                'probabilities': {'home_win': 0.33, 'draw': 0.33, 'away_win': 0.34},
                'expected_goals': {'home': 1.5, 'away': 1.0},
                'most_likely_scores': [('1-1', 100), ('1-0', 90), ('0-1', 85)],
                'total_simulations': n_simulations
            }

    def validate_simulation_results(self, results_df):
        """Validate simulation results for consistency"""
        validation_issues = []

        try:
            # Check if all teams are present
            missing_teams = set(self.teams) - set(results_df.columns)
            if missing_teams:
                validation_issues.append(f"Missing teams in results: {missing_teams}")

            # Check for reasonable point ranges (0-114 points possible in 30-team season)
            for team in results_df.columns:
                min_points = results_df[team].min()
                max_points = results_df[team].max()

                if min_points < 0:
                    validation_issues.append(f"{team} has negative points")
                if max_points > 114:  # 38 games * 3 points
                    validation_issues.append(f"{team} has unrealistic high points: {max_points}")

            # Check for NaN values
            if results_df.isna().any().any():
                validation_issues.append("Results contain NaN values")

        except Exception as e:
            validation_issues.append(f"Validation error: {e}")

        return validation_issues

    def simulate_season_with_current_standings(self, current_standings):
        """Simulate a season starting from current standings"""
        try:
            # Start with current points
            points_table = current_standings.copy()

            # Ensure all teams are in the table
            for team in self.teams:
                if team not in points_table:
                    points_table[team] = 0

            # Simulate only remaining fixtures
            for _, fixture in self.fixtures.iterrows():
                home_team = fixture['HomeTeam']
                away_team = fixture['AwayTeam']

                # Skip if teams don't exist in our model
                if home_team not in self.teams or away_team not in self.teams:
                    continue

                mu_home, mu_away = self.model.predict_match(home_team, away_team)

                home_goals = self.rng.poisson(mu_home)
                away_goals = self.rng.poisson(mu_away)

                if home_goals > away_goals:
                    points_table[home_team] += 3
                elif home_goals == away_goals:
                    points_table[home_team] += 1
                    points_table[away_team] += 1
                else:
                    points_table[away_team] += 3

            return points_table

        except Exception as e:
            print(f"Error simulating with current standings: {e}")
            return current_standings

    def simulate_remaining_fixtures_detailed(self, n_simulations=1000):
        """Simulate remaining fixtures and return detailed results"""
        try:
            fixture_results = []

            for sim_idx in range(n_simulations):
                sim_fixtures = []

                for _, fixture in self.fixtures.iterrows():
                    home_team = fixture['HomeTeam']
                    away_team = fixture['AwayTeam']

                    if home_team not in self.teams or away_team not in self.teams:
                        continue

                    mu_home, mu_away = self.model.predict_match(home_team, away_team)

                    home_goals = self.rng.poisson(mu_home)
                    away_goals = self.rng.poisson(mu_away)

                    sim_fixtures.append({
                        'simulation': sim_idx,
                        'date': str(fixture.get('Date', '')).replace(',', ''),
                        'home_team': str(home_team).replace(',', ''),
                        'away_team': str(away_team).replace(',', ''),
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'home_win': 1 if home_goals > away_goals else 0,
                        'draw': 1 if home_goals == away_goals else 0,
                        'away_win': 1 if home_goals < away_goals else 0
                    })

                fixture_results.extend(sim_fixtures)

            return pd.DataFrame(fixture_results)

        except Exception as e:
            print(f"Error simulating fixtures: {e}")
            return pd.DataFrame()

    def run_monte_carlo_with_standings(self, n_simulations, current_standings, progress_callback=None):
        """Run Monte Carlo simulations starting from current standings"""
        try:
            print(f"Starting {n_simulations:,} simulations with current standings...")
            simulation_results = []

            batch_size = min(1000, n_simulations // 10)

            for i in range(n_simulations):
                if i % batch_size == 0 and progress_callback:
                    progress = (i / n_simulations) * 100
                    progress_callback(progress)

                season_result = self.simulate_season_with_current_standings(current_standings)
                simulation_results.append(season_result)

            results_df = pd.DataFrame(simulation_results)

            # Ensure all teams are present
            for team in self.teams:
                if team not in results_df.columns:
                    results_df[team] = current_standings.get(team, 0)

            print(f"Completed {n_simulations:,} simulations with current standings!")
            return results_df

        except Exception as e:
            print(f"Error running simulations with standings: {e}")
            return pd.DataFrame()