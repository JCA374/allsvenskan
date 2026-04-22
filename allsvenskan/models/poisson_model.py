import pandas as pd
import numpy as np
import pickle
from datetime import datetime
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
        self.last_trained = None
        self.training_window = None  # number of seasons used for training (None = all)

    def fit(self, results_df, team_stats_df):
        """Enhanced fitting with proper MLE and validation - same signature as original"""
        try:
            if results_df.empty or team_stats_df.empty:
                raise ValueError("Empty input data")

            # Parse and sort by date for time-based analysis
            if 'Date' in results_df.columns:
                results_df = results_df.copy()
                results_df['Date'] = pd.to_datetime(results_df['Date'], errors='coerce')
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
            self.last_trained = datetime.now()

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
        """Proper Maximum Likelihood Estimation with vectorized log-likelihood"""
        teams = list(self.attack_rates.keys())
        n_teams = len(teams)
        team_to_idx = {team: i for i, team in enumerate(teams)}

        # Pre-compute index arrays for vectorized likelihood
        valid_mask = (results_df['HomeTeam'].isin(team_to_idx) &
                      results_df['AwayTeam'].isin(team_to_idx))
        valid_df = results_df[valid_mask].reset_index(drop=True)
        home_idx_arr = valid_df['HomeTeam'].map(team_to_idx).values
        away_idx_arr = valid_df['AwayTeam'].map(team_to_idx).values
        home_goals_arr = valid_df['FTHG'].values.astype(int)
        away_goals_arr = valid_df['FTAG'].values.astype(int)

        # Parameter vector: [attack_rates, defense_rates, home_advantage]
        initial_params = np.concatenate(
            [[self.attack_rates[team] for team in teams],
             [self.defense_rates[team] for team in teams],
             [self.home_advantage]])

        # L2 regularization strength — anchors attack/defense near 1.0, resolving
        # the scale invariance: multiplying all attack by c and defense by 1/c
        # leaves the likelihood unchanged but raises the regularization penalty.
        lam = 0.01 * len(valid_df)

        def negative_log_likelihood(params):
            attack_rates = np.clip(params[:n_teams], 0.1, 3.0)
            defense_rates = np.clip(params[n_teams:2 * n_teams], 0.1, 3.0)
            home_adv = np.clip(params[-1], 0.8, 2.5)

            mu_home = np.clip(
                self.league_avg * attack_rates[home_idx_arr] * defense_rates[away_idx_arr] * home_adv,
                0.1, 10.0)
            mu_away = np.clip(
                self.league_avg * attack_rates[away_idx_arr] * defense_rates[home_idx_arr],
                0.1, 10.0)

            log_probs = (poisson.logpmf(home_goals_arr, mu_home) +
                         poisson.logpmf(away_goals_arr, mu_away))
            reg = lam * (np.sum((attack_rates - 1.0) ** 2) + np.sum((defense_rates - 1.0) ** 2))
            return -np.sum(np.clip(log_probs, -10, None)) + reg

        bounds = [(0.1, 3.0)] * (2 * n_teams) + [(0.8, 2.5)]

        try:
            result = optimize.minimize(negative_log_likelihood,
                                       initial_params,
                                       method='L-BFGS-B',
                                       bounds=bounds,
                                       options={'maxiter': 200, 'disp': False})

            if result.success:
                optimized_params = result.x

                # Count actual games per team for shrinkage (not time-weighted —
                # shrinkage is about data quantity, not recency).
                game_counts = {}
                for _, row in results_df.iterrows():
                    game_counts[row['HomeTeam']] = game_counts.get(row['HomeTeam'], 0) + 1
                    game_counts[row['AwayTeam']] = game_counts.get(row['AwayTeam'], 0) + 1

                # Bayesian shrinkage toward league average (1.0).
                # prior_strength = 10 games means a team needs ~10 games before the MLE
                # estimate is trusted more than the prior. With 1 game the estimate is
                # shrunk ~91% toward 1.0; with 30 games it is only ~25% shrunk.
                prior_strength = 10.0
                for i, team in enumerate(teams):
                    raw_atk = max(0.1, optimized_params[i])
                    raw_def = max(0.1, optimized_params[n_teams + i])
                    n_games = game_counts.get(team, 0)
                    shrink = n_games / (n_games + prior_strength)
                    self.attack_rates[team] = shrink * raw_atk + (1 - shrink) * 1.0
                    self.defense_rates[team] = shrink * raw_def + (1 - shrink) * 1.0

                self.home_advantage = float(np.clip(optimized_params[-1], 0.8, 2.5))
                print(f"✅ MLE optimization converged. Final log-likelihood: {-result.fun:.2f}")
            else:
                print("⚠️ MLE optimization did not converge. Using refined parameters.")

        except Exception as e:
            print(f"⚠️ MLE optimization error: {e}. Using refined parameters.")

    def _estimate_correlation(self, results_df):
        """Estimate Dixon-Coles correlation parameter for low-scoring games.

        Computes expected 0-0 and 1-1 rates from the fitted Poisson parameters
        rather than using hardcoded league-agnostic constants.
        """
        try:
            total_matches = len(results_df)
            if total_matches < 10:
                self.rho = 0.0
                return

            # Count observed low-scoring outcomes
            observed_00 = int(((results_df['FTHG'] == 0) & (results_df['FTAG'] == 0)).sum())
            observed_11 = int(((results_df['FTHG'] == 1) & (results_df['FTAG'] == 1)).sum())

            observed_00_rate = observed_00 / total_matches
            observed_11_rate = observed_11 / total_matches

            # Compute expected rates from the fitted Poisson parameters per match
            exp_00_list = []
            exp_11_list = []
            for _, row in results_df.iterrows():
                home = row['HomeTeam']
                away = row['AwayTeam']
                if home not in self.attack_rates or away not in self.attack_rates:
                    continue
                mu_h = self.league_avg * self.attack_rates[home] * self.defense_rates[away] * self.home_advantage
                mu_a = self.league_avg * self.attack_rates[away] * self.defense_rates[home]
                mu_h = max(0.1, mu_h)
                mu_a = max(0.1, mu_a)
                exp_00_list.append(poisson.pmf(0, mu_h) * poisson.pmf(0, mu_a))
                exp_11_list.append(poisson.pmf(1, mu_h) * poisson.pmf(1, mu_a))

            if not exp_00_list:
                self.rho = 0.0
                return

            expected_00_rate = float(np.mean(exp_00_list))
            expected_11_rate = float(np.mean(exp_11_list))

            self.rho = float(np.clip(
                (observed_00_rate - expected_00_rate) + (observed_11_rate - expected_11_rate),
                -0.2, 0.2,
            ))

        except Exception as e:
            self.rho = 0.0

    def _dixon_coles_adjustment(self, home_goals, away_goals, mu_home,
                                mu_away):
        """Dixon-Coles adjustment for low-scoring games"""
        if self.rho == 0:
            return 1.0

        if home_goals == 0 and away_goals == 0:
            factor = 1 - mu_home * mu_away * self.rho
        elif home_goals == 0 and away_goals == 1:
            factor = 1 + mu_home * self.rho
        elif home_goals == 1 and away_goals == 0:
            factor = 1 + mu_away * self.rho
        elif home_goals == 1 and away_goals == 1:
            factor = 1 - self.rho
        else:
            return 1.0

        # Guard against extreme mu/rho combinations producing invalid factors
        if not (0.01 <= factor <= 10.0):
            return 1.0
        return factor

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
                weights = np.exp(-self.time_decay * days_from_recent).values
            else:
                weights = np.ones(len(results_df))

            home_team_arr = results_df['HomeTeam'].values
            away_team_arr = results_df['AwayTeam'].values

            for team in self.attack_rates.keys():
                home_mask = home_team_arr == team
                away_mask = away_team_arr == team
                home_matches = results_df[home_mask]
                away_matches = results_df[away_mask]

                # Compute attack estimates from home and away separately, then average
                attack_estimates = []
                if len(home_matches) > 0:
                    home_attack = np.average(
                        home_matches['FTHG'], weights=weights[home_mask]
                    ) / self.league_avg
                    attack_estimates.append(home_attack)
                if len(away_matches) > 0:
                    # Scale away goals up by home_advantage to put on same scale as home attack
                    away_attack = np.average(
                        away_matches['FTAG'], weights=weights[away_mask]
                    ) / self.league_avg * self.home_advantage
                    attack_estimates.append(away_attack)

                if attack_estimates:
                    data_attack = np.mean(attack_estimates)
                    self.attack_rates[team] = max(
                        0.1, (self.attack_rates[team] + data_attack) / 2)

                # Defense rate (goals conceded relative to league avg)
                if len(home_matches) > 0:
                    home_defense = np.average(
                        home_matches['FTAG'], weights=weights[home_mask]
                    ) / self.league_avg
                else:
                    home_defense = 1.0

                if len(away_matches) > 0:
                    away_defense = np.average(
                        away_matches['FTHG'], weights=weights[away_mask]
                    ) / self.league_avg / self.home_advantage
                else:
                    away_defense = 1.0

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
                'validation_score': self.validation_score,
                'last_trained': self.last_trained,
                'training_window': self.training_window,
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
            self.last_trained = model_data.get('last_trained', None)
            self.training_window = model_data.get('training_window', None)

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
            'last_trained':
            self.last_trained.strftime('%Y-%m-%d %H:%M')
            if self.last_trained else 'Unknown',
            'strongest_attack':
            max(self.attack_rates.items(), key=lambda x: x[1])
            if self.attack_rates else None,
            'strongest_defense':
            min(self.defense_rates.items(), key=lambda x: x[1])
            if self.defense_rates else None
        }

        return summary

    @staticmethod
    def walk_forward_cv(results_df, lookback_windows=None, current_season_val_frac=0.4):
        """Walk-forward cross-validation across different historical lookback windows.

        Runs two types of evaluation:

        1. **Historical CV** — for each window k, trains on the k completed seasons
           before the most-recent completed season, validates on that last completed
           season. Validates model accuracy on unseen full-season data.

        2. **Current-season split** — splits the in-progress season at
           ``(1 - current_season_val_frac)`` of its matches. Trains on k prior
           seasons + the early part of the current season; validates on the held-out
           tail. This directly measures how well the model predicts remaining matches
           this season as form data accumulates.

        Why differences between windows are small
        ------------------------------------------
        The model applies exponential time-decay (time_decay=0.01). A match played
        365 days ago is weighted e^{-3.65} ≈ 0.026 vs 1.0 for today. Matches from
        2+ years ago weigh < 0.07 % and contribute almost nothing to the fitted
        parameters. Adding more historical seasons therefore barely moves log-loss —
        the effect is real but small by design.

        Args:
            results_df: Full multi-season results DataFrame with 'SeasonStart' column.
            lookback_windows: List of ints — number of past seasons to train on.
                              Defaults to [1, 2, 3, 5, 8].
            current_season_val_frac: Fraction of current-season matches held out for
                                     validation (default 0.4 = last 40%).

        Returns:
            dict with keys:
              'historical'      — list of result dicts (historical CV)
              'current_season'  — list of result dicts (in-progress season split),
                                  or empty list if < 10 current-season matches exist.
        """
        from allsvenskan.data.strength import TeamStrengthCalculator

        if lookback_windows is None:
            lookback_windows = [1, 2, 3, 5, 8]

        if 'SeasonStart' not in results_df.columns:
            raise ValueError("results_df must have a 'SeasonStart' column")

        seasons = sorted(results_df['SeasonStart'].dropna().astype(int).unique())
        if len(seasons) < 2:
            raise ValueError("Need at least 2 seasons for walk-forward CV")

        def _eval_log_loss(model, val_df):
            vals = []
            for _, row in val_df.iterrows():
                try:
                    probs = model.predict_outcome_probabilities(row['HomeTeam'], row['AwayTeam'])
                    hg, ag = int(row['FTHG']), int(row['FTAG'])
                    p = probs['home_win'] if hg > ag else (probs['draw'] if hg == ag else probs['away_win'])
                    vals.append(-np.log(max(1e-6, p)))
                except Exception:
                    continue
            return vals

        def _train_and_eval(train_df, val_df, train_seasons):
            if len(train_df) < 20:
                return None
            try:
                strength_calc = TeamStrengthCalculator(use_odds_integration=False)
                team_stats = strength_calc.calculate_strengths(train_df)
                model = PoissonModel(use_mle=False, use_dixon_coles=False)
                model.fit(train_df, team_stats)
                if not model.fitted:
                    return None
                vals = _eval_log_loss(model, val_df)
                if not vals:
                    return None
                return {
                    'train_seasons': train_seasons,
                    'n_train': len(train_df),
                    'n_val': len(vals),
                    'log_loss': float(np.mean(vals)),
                }
            except Exception:
                return None

        # ── 1. Historical CV ──────────────────────────────────────────────────
        current_season = seasons[-1]
        completed_seasons = seasons[:-1]  # exclude the in-progress season
        val_season = completed_seasons[-1] if completed_seasons else current_season
        val_df_hist = results_df[results_df['SeasonStart'] == val_season].copy()

        historical = []
        prior_seasons = [s for s in completed_seasons if s != val_season]
        for k in lookback_windows:
            train_seasons = prior_seasons[-k:] if k <= len(prior_seasons) else prior_seasons
            if not train_seasons:
                continue
            train_df = results_df[results_df['SeasonStart'].isin(train_seasons)].copy()
            row = _train_and_eval(train_df, val_df_hist, train_seasons)
            if row:
                historical.append({'lookback': k, **row})

        # ── 2. Current-season split ───────────────────────────────────────────
        current_df = results_df[results_df['SeasonStart'] == current_season].copy()
        if 'Date' in current_df.columns:
            current_df = current_df.sort_values('Date')

        current_season_results = []
        n_current = len(current_df)
        split_idx = int(n_current * (1 - current_season_val_frac))

        if split_idx >= 10 and (n_current - split_idx) >= 5:
            cur_train_part = current_df.iloc[:split_idx]
            cur_val_part   = current_df.iloc[split_idx:]

            for k in lookback_windows:
                prior = prior_seasons[-k:] if k <= len(prior_seasons) else prior_seasons
                hist_part = results_df[results_df['SeasonStart'].isin(prior)].copy() if prior else pd.DataFrame()
                train_df = pd.concat([hist_part, cur_train_part], ignore_index=True) if not hist_part.empty else cur_train_part.copy()
                row = _train_and_eval(train_df, cur_val_part, (prior or []) + [current_season])
                if row:
                    current_season_results.append({'lookback': k, **row})

        return {
            'historical': historical,
            'current_season': current_season_results,
        }
