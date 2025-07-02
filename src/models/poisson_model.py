import pandas as pd
import numpy as np
import pickle
from scipy import optimize
from scipy.stats import poisson

class PoissonModel:
    def __init__(self):
        self.attack_rates = {}
        self.defense_rates = {}
        self.home_advantage = 1.0
        self.league_avg = 1.0
        self.fitted = False
    
    def fit(self, results_df, team_stats_df):
        """Fit Poisson model parameters from historical data"""
        try:
            if results_df.empty or team_stats_df.empty:
                raise ValueError("Empty input data")
            
            # Calculate league averages
            total_goals = results_df['FTHG'].sum() + results_df['FTAG'].sum()
            total_matches = len(results_df)
            self.league_avg = total_goals / (total_matches * 2) if total_matches > 0 else 1.0
            
            # Initialize team parameters
            teams = list(set(results_df['HomeTeam'].unique()) | set(results_df['AwayTeam'].unique()))
            
            # Use team stats to initialize attack and defense rates
            for team in teams:
                if team in team_stats_df.index:
                    stats = team_stats_df.loc[team]
                    self.attack_rates[team] = max(0.1, stats.get('attack_strength', 1.0))
                    self.defense_rates[team] = max(0.1, stats.get('defense_strength', 1.0))
                else:
                    self.attack_rates[team] = 1.0
                    self.defense_rates[team] = 1.0
            
            # Estimate home advantage
            self.home_advantage = self._estimate_home_advantage(results_df)
            
            # Optionally refine parameters using MLE
            self._refine_parameters(results_df)
            
            self.fitted = True
            
        except Exception as e:
            print(f"Error fitting Poisson model: {e}")
            self._set_default_parameters(results_df)
    
    def _estimate_home_advantage(self, results_df):
        """Estimate home advantage factor"""
        try:
            home_goals = results_df['FTHG'].mean()
            away_goals = results_df['FTAG'].mean()
            
            if away_goals > 0:
                return max(1.0, home_goals / away_goals)
            else:
                return 1.3  # Default home advantage
                
        except Exception as e:
            return 1.3
    
    def _refine_parameters(self, results_df):
        """Refine parameters using maximum likelihood estimation"""
        try:
            # This is a simplified approach - in practice, you might use
            # more sophisticated optimization methods
            
            # Calculate empirical rates for each team
            for team in self.attack_rates.keys():
                # Home attack rate
                home_matches = results_df[results_df['HomeTeam'] == team]
                if len(home_matches) > 0:
                    home_attack = home_matches['FTHG'].mean() / self.league_avg
                    self.attack_rates[team] = max(0.1, (self.attack_rates[team] + home_attack) / 2)
                
                # Away attack rate (adjust for home advantage)
                away_matches = results_df[results_df['AwayTeam'] == team]
                if len(away_matches) > 0:
                    away_attack = away_matches['FTAG'].mean() / self.league_avg * self.home_advantage
                    self.attack_rates[team] = max(0.1, (self.attack_rates[team] + away_attack) / 2)
                
                # Defense rate (goals conceded)
                home_defense = home_matches['FTAG'].mean() / self.league_avg if len(home_matches) > 0 else self.league_avg
                away_defense = away_matches['FTHG'].mean() / self.league_avg / self.home_advantage if len(away_matches) > 0 else self.league_avg
                
                self.defense_rates[team] = max(0.1, (home_defense + away_defense) / 2)
                
        except Exception as e:
            print(f"Error refining parameters: {e}")
    
    def _set_default_parameters(self, results_df):
        """Set default parameters if fitting fails"""
        teams = list(set(results_df['HomeTeam'].unique()) | set(results_df['AwayTeam'].unique()))
        
        for team in teams:
            self.attack_rates[team] = 1.0
            self.defense_rates[team] = 1.0
        
        self.home_advantage = 1.3
        self.league_avg = 1.5
        self.fitted = True
    
    def predict_match(self, home_team, away_team):
        """Predict match outcome probabilities"""
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
        """Calculate win/draw/loss probabilities"""
        try:
            mu_home, mu_away = self.predict_match(home_team, away_team)
            
            prob_home_win = 0
            prob_draw = 0
            prob_away_win = 0
            
            for home_goals in range(max_goals + 1):
                for away_goals in range(max_goals + 1):
                    prob = poisson.pmf(home_goals, mu_home) * poisson.pmf(away_goals, mu_away)
                    
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
        """Save model parameters to file"""
        try:
            model_data = {
                'attack_rates': self.attack_rates,
                'defense_rates': self.defense_rates,
                'home_advantage': self.home_advantage,
                'league_avg': self.league_avg,
                'fitted': self.fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load(self, filepath):
        """Load model parameters from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.attack_rates = model_data.get('attack_rates', {})
            self.defense_rates = model_data.get('defense_rates', {})
            self.home_advantage = model_data.get('home_advantage', 1.3)
            self.league_avg = model_data.get('league_avg', 1.5)
            self.fitted = model_data.get('fitted', False)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.fitted = False
    
    def get_model_summary(self):
        """Get summary of model parameters"""
        if not self.fitted:
            return "Model not fitted"
        
        summary = {
            'teams_count': len(self.attack_rates),
            'home_advantage': round(self.home_advantage, 3),
            'league_avg_goals': round(self.league_avg, 3),
            'strongest_attack': max(self.attack_rates.items(), key=lambda x: x[1]) if self.attack_rates else None,
            'strongest_defense': min(self.defense_rates.items(), key=lambda x: x[1]) if self.defense_rates else None
        }
        
        return summary
