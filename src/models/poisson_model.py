import pandas as pd
import numpy as np
import pickle
from scipy import optimize
from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import warnings


class PoissonModel:

    def __init__(self, time_decay=0.01, use_mle=False, use_dixon_coles=False):
        """
        Enhanced Poisson model with optional advanced features for faster training

        Args:
            time_decay: Exponential decay factor for time weighting
            use_mle: Whether to use Maximum Likelihood Estimation (slower but more accurate)
            use_dixon_coles: Whether to apply Dixon-Coles correlation adjustment
        """
        self.attack_rates = {}
        self.defense_rates = {}
        self.home_advantage = 1.0
        self.league_avg = 1.0
        self.fitted = False
        self.time_decay = time_decay
        self.use_mle = use_mle
        self.use_dixon_coles = use_dixon_coles
        self.rho = 0.0  # Dixon-Coles correlation parameter
        self.validation_score = None

    def fit(self, results_df, team_stats_df):
        """Enhanced fitting with proper MLE and validation - same signature as original"""
        try:
            if results_df.empty or team_stats_df.empty:
                raise ValueError("Empty input data")

            # Sort by date for time-based analysis if date column exists
            if 'Date' in results_df.columns:
                results_df = results_df.sort_values('Date')

            # Calculate league averages with time weighting
            self.league_avg = self._calculate_weighted_league_avg(results_df)

            # Initialize parameters from team stats
            self._initialize_parameters(results_df, team_stats_df)

            # Estimate home advantage
            self.home_advantage = self._estimate_home_advantage(results_df)

            # Use fast parameter refinement by default
            self._refine_parameters(results_df)
            
            # Only use MLE if explicitly requested and sufficient data
            if self.use_mle and len(results_df) > 100:
                print("Using MLE optimization (this may take longer)...")
                self._fit_mle(results_df)

            # Only use Dixon-Coles if explicitly requested
            if self.use_dixon_coles:
                self._estimate_correlation(results_df)

            # Skip validation by default for faster training
            # Uncomment below for validation if needed
            # if len(results_df) > 100:
            #     self.validation_score = self._cross_validate(results_df)
            #     print(f"✅ Model validation log-loss: {self.validation_score:.4f}")

            self.fitted = True

        except Exception as e:
            print(f"Error fitting enhanced model: {e}")
            self._set_default_parameters(results_df)

    def _calculate_weighted_league_avg(self, results_df):
        """Calculate league average with exponential time weighting"""
        try:
            if 'Date' not in results_df.columns or len(results_df) == 0:
                # Fallback to simple average
                total_goals = results_df['FTHG'].sum(
                ) + results_df['FTAG'].sum()
                total_matches = len(results_df)
                return total_goals / (total_matches *
                                      2) if total_matches > 0 else 1.4

            # More recent matches get higher weights
            max_date = results_df['Date'].max()
            days_from_recent = (max_date - results_df['Date']).dt.days
            weights = np.exp(-self.time_decay * days_from_recent)

            total_goals = results_df['FTHG'] + results_df['FTAG']
            weighted_avg = np.average(total_goals, weights=weights) / 2

            return max(0.5, weighted_avg)

        except Exception as e:
            # Fallback calculation
            total_goals = results_df['FTHG'].sum() + results_df['FTAG'].sum()
            total_matches = len(results_df)
            return total_goals / (total_matches *
                                  2) if total_matches > 0 else 1.4

    def _initialize_parameters(self, results_df, team_stats_df):
        """Initialize parameters with improved team strength calculation"""
        teams = list(
            set(results_df['HomeTeam'].unique())
            | set(results_df['AwayTeam'].unique()))

        for team in teams:
            if team in team_stats_df.index:
                stats = team_stats_df.loc[team]
                # Use enhanced team stats with time weighting
                self.attack_rates[team] = max(
                    0.3, stats.get('attack_strength', 1.0))
                self.defense_rates[team] = max(
                    0.3, stats.get('defense_strength', 1.0))
            else:
                # Default values for teams with insufficient data
                self.attack_rates[team] = 1.0
                self.defense_rates[team] = 1.0

    def _fit_mle(self, results_df):
        """Proper Maximum Likelihood Estimation"""
        teams = list(self.attack_rates.keys())
        n_teams = len(teams)
        team_to_idx = {team: i for i, team in enumerate(teams)}

        # Parameter vector: [attack_rates, defense_rates, home_advantage]
        initial_params = np.concatenate(
            [[self.attack_rates[team] for team in teams],
             [self.defense_rates[team] for team in teams],
             [self.home_advantage]])

        def negative_log_likelihood(params):
            """Calculate negative log-likelihood for optimization"""
            try:
                attack_rates = params[:n_teams]
                defense_rates = params[n_teams:2 * n_teams]
                home_adv = params[-1]

                # Ensure positive parameters
                attack_rates = np.maximum(0.1, attack_rates)
                defense_rates = np.maximum(0.1, defense_rates)
                home_adv = max(0.8, min(2.5, home_adv))

                log_likelihood = 0

                for _, match in results_df.iterrows():
                    if match['HomeTeam'] not in team_to_idx or match[
                            'AwayTeam'] not in team_to_idx:
                        continue

                    home_idx = team_to_idx[match['HomeTeam']]
                    away_idx = team_to_idx[match['AwayTeam']]

                    # Expected goals
                    mu_home = self.league_avg * attack_rates[
                        home_idx] * defense_rates[away_idx] * home_adv
                    mu_away = self.league_avg * attack_rates[
                        away_idx] * defense_rates[home_idx]

                    # Ensure positive values
                    mu_home = max(0.1, min(10.0, mu_home))
                    mu_away = max(0.1, min(10.0, mu_away))

                    # Add to log-likelihood
                    home_goals = int(match['FTHG'])
                    away_goals = int(match['FTAG'])

                    # Basic Poisson probability
                    prob = (poisson.pmf(home_goals, mu_home) *
                            poisson.pmf(away_goals, mu_away))

                    # Apply Dixon-Coles adjustment if enabled
                    if self.use_dixon_coles:
                        prob *= self._dixon_coles_adjustment(
                            home_goals, away_goals, mu_home, mu_away)

                    if prob > 1e-15:  # Avoid log(0)
                        log_likelihood += np.log(prob)
                    else:
                        log_likelihood -= 10  # Penalty for very unlikely events

                return -log_likelihood

            except Exception as e:
                return 1e6  # Large penalty for errors

        # Constraints: all parameters must be positive
        bounds = [(0.1, 3.0)] * (2 * n_teams) + [(0.8, 2.5)
                                                 ]  # home advantage bounds

        try:
            # Use single fast optimization method
            result = optimize.minimize(negative_log_likelihood,
                                     initial_params,
                                     method='L-BFGS-B',
                                     bounds=bounds,
                                     options={
                                         'maxiter': 200,  # Reduced iterations
                                         'disp': False
                                     })

            if result.success:
                # Update parameters with optimized values
                optimized_params = result.x
                for i, team in enumerate(teams):
                    self.attack_rates[team] = max(0.1, optimized_params[i])
                    self.defense_rates[team] = max(
                        0.1, optimized_params[n_teams + i])
                self.home_advantage = max(0.8, min(2.5, optimized_params[-1]))

                print(
                    f"✅ MLE optimization converged. Final log-likelihood: {-result.fun:.2f}"
                )
            else:
                print(
                    "⚠️ MLE optimization failed. Using refined initial parameters."
                )
                self._refine_parameters(results_df)

        except Exception as e:
            print(f"⚠️ MLE optimization error: {e}. Using refined parameters.")
            self._refine_parameters(results_df)

    def _estimate_correlation(self, results_df):
        """Estimate Dixon-Coles correlation parameter for low-scoring games"""
        try:
            # Focus on low-scoring games (0-0, 1-0, 0-1, 1-1)
            low_scoring = results_df[((results_df['FTHG'] <= 1) &
                                      (results_df['FTAG'] <= 1))]

            if len(low_scoring) > 10:
                # Count specific low-scoring outcomes
                total_matches = len(results_df)
                observed_00 = len(low_scoring[(low_scoring['FTHG'] == 0)
                                              & (low_scoring['FTAG'] == 0)])
                observed_11 = len(low_scoring[(low_scoring['FTHG'] == 1)
                                              & (low_scoring['FTAG'] == 1)])
                observed_01 = len(low_scoring[(low_scoring['FTHG'] == 0)
                                              & (low_scoring['FTAG'] == 1)])
                observed_10 = len(low_scoring[(low_scoring['FTHG'] == 1)
                                              & (low_scoring['FTAG'] == 0)])

                # Simple correlation estimate based on observed vs expected ratios
                expected_00_rate = 0.08  # Typical rate for 0-0 draws
                expected_11_rate = 0.12  # Typical rate for 1-1 draws

                observed_00_rate = observed_00 / total_matches
                observed_11_rate = observed_11 / total_matches

                # Rough correlation estimate
                self.rho = min(
                    0.2,
                    max(-0.2, (observed_00_rate - expected_00_rate) +
                        (observed_11_rate - expected_11_rate)))
            else:
                self.rho = 0.0

        except Exception as e:
            self.rho = 0.0

    def _dixon_coles_adjustment(self, home_goals, away_goals, mu_home,
                                mu_away):
        """Dixon-Coles adjustment for low-scoring games"""
        if self.rho == 0:
            return 1.0

        if home_goals == 0 and away_goals == 0:
            return 1 - mu_home * mu_away * self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + mu_home * self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + mu_away * self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - self.rho
        else:
            return 1.0

    def _cross_validate(self, results_df, n_splits=3):
        """Time-series cross-validation with enhanced error handling"""
        try:
            if len(results_df) < 50:
                return None

            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []

            for train_idx, test_idx in tscv.split(results_df):
                try:
                    train_data = results_df.iloc[train_idx]
                    test_data = results_df.iloc[test_idx]

                    if len(train_data) < 20 or len(test_data) < 5:
                        continue

                    # Create temporary model for this fold
                    temp_model = PoissonModel(self.time_decay,
                                              use_mle=False,
                                              use_dixon_coles=False)
                    temp_model.league_avg = temp_model._calculate_weighted_league_avg(
                        train_data)

                    # Simple initialization for validation
                    teams = list(
                        set(train_data['HomeTeam'].unique())
                        | set(train_data['AwayTeam'].unique()))
                    for team in teams:
                        temp_model.attack_rates[team] = 1.0
                        temp_model.defense_rates[team] = 1.0

                    temp_model.home_advantage = temp_model._estimate_home_advantage(
                        train_data)
                    temp_model._refine_parameters(train_data)
                    temp_model.fitted = True

                    # Predict on test data
                    predictions = []
                    actuals = []

                    for _, match in test_data.iterrows():
                        try:
                            prob_dist = temp_model.predict_outcome_probabilities(
                                match['HomeTeam'], match['AwayTeam'])

                            # Convert actual result to categorical
                            if match['FTHG'] > match['FTAG']:
                                actual = [1, 0, 0]  # Home win
                            elif match['FTHG'] < match['FTAG']:
                                actual = [0, 0, 1]  # Away win
                            else:
                                actual = [0, 1, 0]  # Draw

                            predicted = [
                                prob_dist['home_win'], prob_dist['draw'],
                                prob_dist['away_win']
                            ]

                            predictions.append(predicted)
                            actuals.append(actual)

                        except Exception as e:
                            continue

                    # Calculate log-loss for this fold
                    if len(predictions) > 0:
                        score = log_loss(actuals, predictions)
                        scores.append(score)

                except Exception as e:
                    continue

            return np.mean(scores) if scores else None

        except Exception as e:
            print(f"Cross-validation error: {e}")
            return None

    def _estimate_home_advantage(self, results_df):
        """Improved home advantage estimation with time weighting"""
        try:
            if 'Date' in results_df.columns and len(results_df) > 10:
                # Weight recent matches more heavily
                max_date = results_df['Date'].max()
                days_from_recent = (max_date - results_df['Date']).dt.days
                weights = np.exp(-self.time_decay * days_from_recent)

                home_goals = np.average(results_df['FTHG'], weights=weights)
                away_goals = np.average(results_df['FTAG'], weights=weights)
            else:
                home_goals = results_df['FTHG'].mean()
                away_goals = results_df['FTAG'].mean()

            if away_goals > 0:
                advantage = home_goals / away_goals
                return max(1.0, min(2.0, advantage))
            else:
                return 1.3

        except Exception as e:
            return 1.3

    def _refine_parameters(self, results_df):
        """Enhanced parameter refinement using weighted averages"""
        try:
            # Calculate time weights if available
            if 'Date' in results_df.columns:
                max_date = results_df['Date'].max()
                days_from_recent = (max_date - results_df['Date']).dt.days
                weights = np.exp(-self.time_decay * days_from_recent)
            else:
                weights = np.ones(len(results_df))

            # Refine each team's parameters
            for team in self.attack_rates.keys():
                # Home attack rate
                home_matches = results_df[results_df['HomeTeam'] == team]
                if len(home_matches) > 0:
                    home_weights = weights[results_df['HomeTeam'] == team]
                    home_attack = np.average(
                        home_matches['FTHG'],
                        weights=home_weights) / self.league_avg
                    self.attack_rates[team] = max(
                        0.1, (self.attack_rates[team] + home_attack) / 2)

                # Away attack rate (adjust for home advantage)
                away_matches = results_df[results_df['AwayTeam'] == team]
                if len(away_matches) > 0:
                    away_weights = weights[results_df['AwayTeam'] == team]
                    away_attack = np.average(
                        away_matches['FTAG'], weights=away_weights
                    ) / self.league_avg * self.home_advantage
                    self.attack_rates[team] = max(
                        0.1, (self.attack_rates[team] + away_attack) / 2)

                # Defense rate (goals conceded)
                if len(home_matches) > 0:
                    home_weights = weights[results_df['HomeTeam'] == team]
                    home_defense = np.average(
                        home_matches['FTAG'],
                        weights=home_weights) / self.league_avg
                else:
                    home_defense = self.league_avg

                if len(away_matches) > 0:
                    away_weights = weights[results_df['AwayTeam'] == team]
                    away_defense = np.average(
                        away_matches['FTHG'], weights=away_weights
                    ) / self.league_avg / self.home_advantage
                else:
                    away_defense = self.league_avg

                self.defense_rates[team] = max(
                    0.1, (home_defense + away_defense) / 2)

        except Exception as e:
            print(f"Error refining parameters: {e}")

    def _set_default_parameters(self, results_df):
        """Set reasonable defaults if all else fails"""
        teams = list(
            set(results_df['HomeTeam'].unique())
            | set(results_df['AwayTeam'].unique()))

        for team in teams:
            self.attack_rates[team] = 1.0
            self.defense_rates[team] = 1.0

        self.home_advantage = 1.3
        self.league_avg = 1.5
        self.fitted = True

    def predict_match(self, home_team, away_team):
        """Predict match outcome probabilities - same signature as original"""
        try:
            if not self.fitted:
                raise ValueError("Model not fitted")

            # Get team parameters (use defaults if team not found)
            home_attack = self.attack_rates.get(home_team, 1.0)
            home_defense = self.defense_rates.get(home_team, 1.0)
            away_attack = self.attack_rates.get(away_team, 1.0)
            away_defense = self.defense_rates.get(away_team, 1.0)

            # Calculate expected goals
            mu_home = self.league_avg * home_attack * away_defense * self.home_advantage
            mu_away = self.league_avg * away_attack * home_defense

            # Ensure positive values
            mu_home = max(0.1, mu_home)
            mu_away = max(0.1, mu_away)

            return mu_home, mu_away

        except Exception as e:
            print(f"Error predicting match: {e}")
            return 1.5, 1.0  # Default values

    def predict_outcome_probabilities(self, home_team, away_team, max_goals=6):
        """Calculate win/draw/loss probabilities - same signature as original"""
        try:
            mu_home, mu_away = self.predict_match(home_team, away_team)

            prob_home_win = 0
            prob_draw = 0
            prob_away_win = 0

            for home_goals in range(max_goals + 1):
                for away_goals in range(max_goals + 1):
                    # Basic Poisson probability
                    prob = poisson.pmf(home_goals, mu_home) * poisson.pmf(
                        away_goals, mu_away)

                    # Apply Dixon-Coles adjustment if enabled
                    if self.use_dixon_coles:
                        prob *= self._dixon_coles_adjustment(
                            home_goals, away_goals, mu_home, mu_away)

                    if home_goals > away_goals:
                        prob_home_win += prob
                    elif home_goals == away_goals:
                        prob_draw += prob
                    else:
                        prob_away_win += prob

            return {
                'home_win': prob_home_win,
                'draw': prob_draw,
                'away_win': prob_away_win,
                'mu_home': mu_home,
                'mu_away': mu_away
            }

        except Exception as e:
            print(f"Error calculating probabilities: {e}")
            return {
                'home_win': 0.33,
                'draw': 0.33,
                'away_win': 0.34,
                'mu_home': 1.5,
                'mu_away': 1.0
            }

    def save(self, filepath):
        """Save model parameters to file - same signature as original"""
        try:
            model_data = {
                'attack_rates': self.attack_rates,
                'defense_rates': self.defense_rates,
                'home_advantage': self.home_advantage,
                'league_avg': self.league_avg,
                'fitted': self.fitted,
                'rho': self.rho,
                'time_decay': self.time_decay,
                'use_mle': self.use_mle,
                'use_dixon_coles': self.use_dixon_coles,
                'validation_score': self.validation_score
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, filepath):
        """Load model parameters from file - same signature as original"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.attack_rates = model_data.get('attack_rates', {})
            self.defense_rates = model_data.get('defense_rates', {})
            self.home_advantage = model_data.get('home_advantage', 1.3)
            self.league_avg = model_data.get('league_avg', 1.5)
            self.fitted = model_data.get('fitted', False)
            self.rho = model_data.get('rho', 0.0)
            self.time_decay = model_data.get('time_decay', 0.01)
            self.use_mle = model_data.get('use_mle', True)
            self.use_dixon_coles = model_data.get('use_dixon_coles', True)
            self.validation_score = model_data.get('validation_score', None)

        except Exception as e:
            print(f"Error loading model: {e}")
            self.fitted = False

    def get_model_summary(self):
        """Get summary of model parameters - same signature as original"""
        if not self.fitted:
            return "Model not fitted"

        summary = {
            'teams_count':
            len(self.attack_rates),
            'home_advantage':
            round(self.home_advantage, 3),
            'league_avg_goals':
            round(self.league_avg, 3),
            'dixon_coles_rho':
            round(self.rho, 3),
            'time_decay':
            self.time_decay,
            'validation_score':
            round(self.validation_score, 4)
            if self.validation_score else 'N/A',
            'strongest_attack':
            max(self.attack_rates.items(), key=lambda x: x[1])
            if self.attack_rates else None,
            'strongest_defense':
            min(self.defense_rates.items(), key=lambda x: x[1])
            if self.defense_rates else None
        }

        return summary
