"""
Team Strength Extraction from Odds

This module implements advanced team strength calculation by reverse-engineering
betting odds to extract implied team attack/defense strengths and expected goals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.stats import poisson
import warnings

from src.data.odds_schema import OddsRecord
from src.utils.odds_converter import remove_margin, calculate_implied_probabilities

logger = logging.getLogger(__name__)

class OddsStrengthExtractor:
    """Extract team attack/defense strengths from betting odds"""
    
    def __init__(self, cache_size: int = 128, max_goals: int = 6):
        self.cache_size = cache_size
        self.max_goals = max_goals
        self._strength_cache = {}
        self._match_cache = {}
        
    def calculate_team_strength(self, team: str, odds_history: List[Dict], 
                               lookback: Optional[int] = None) -> Dict[str, float]:
        """
        Extract attack/defense ratings from odds with caching
        
        Args:
            team: Team name
            odds_history: List of odds records for the team
            lookback: Number of recent matches to use (auto-determined if None)
            
        Returns:
            Dictionary with attack, defense, and form strengths
        """
        # Create cache key
        cache_key = f"{team}_{len(odds_history)}_{lookback}"
        
        if cache_key in self._strength_cache:
            logger.debug(f"Using cached strength for {team}")
            return self._strength_cache[cache_key]
        
        try:
            # Determine optimal lookback if not specified
            if lookback is None:
                lookback = self._find_optimal_lookback(team, odds_history)
            
            # Ensure lookback is valid
            if lookback <= 0:
                lookback = min(5, len(odds_history))
            
            # Get recent games
            recent_games = odds_history[-lookback:] if lookback > 0 else odds_history
            
            # Calculate strengths
            strengths = self._calculate_strengths_from_odds(team, recent_games)
            
            # Cache result
            self._strength_cache[cache_key] = strengths
            
            # Manage cache size
            if len(self._strength_cache) > self.cache_size:
                # Remove oldest entries
                oldest_key = next(iter(self._strength_cache))
                del self._strength_cache[oldest_key]
            
            return strengths
           
        except Exception as e:
            logger.error(f"Error calculating strength for {team}: {e}")
            return {'attack': 1.0, 'defense': 1.0, 'form': 0.5}
    
    def _calculate_strengths_from_odds(self, team: str, games: List[Dict]) -> Dict[str, float]:
        """Core strength calculation with Over/Under integration"""
        if not games:
            return {'attack': 1.0, 'defense': 1.0, 'form': 0.5}
        
        attack_scores = []
        defense_scores = []
        form_scores = []
        weights = []
        
        for i, game in enumerate(games):
            try:
                # Time-based weight (more recent = higher weight)
                weight = np.exp(-0.1 * (len(games) - i - 1))
                weights.append(weight)
                
                # Extract probabilities from odds
                if game.get('is_home', True):
                    win_prob = 1 / game['home_odds']
                    lose_prob = 1 / game['away_odds']
                    draw_prob = 1 / game.get('draw_odds', 3.5)
                else:
                    win_prob = 1 / game['away_odds']
                    lose_prob = 1 / game['home_odds']
                    draw_prob = 1 / game.get('draw_odds', 3.5)
                
                # Normalize probabilities
                total_prob = win_prob + draw_prob + lose_prob
                win_prob /= total_prob
                draw_prob /= total_prob
                lose_prob /= total_prob
                
                # Extract expected goals from Over/Under odds if available
                if 'over_2.5_odds' in game and 'under_2.5_odds' in game:
                    over_prob = 1 / game['over_2.5_odds']
                    under_prob = 1 / game['under_2.5_odds']
                    # Normalize
                    total_ou = over_prob + under_prob
                    over_prob /= total_ou
                    # Expected total goals
                    expected_total_goals = 2.5 + (over_prob - 0.5) * 2
                else:
                    # Use win/draw probabilities to estimate goals
                    expected_total_goals = 2.5 + (win_prob - lose_prob) * 1.5
                
                # Estimate team's contribution to total goals
                team_goal_share = 0.5 + (win_prob - lose_prob) * 0.3
                expected_team_goals = expected_total_goals * team_goal_share
                expected_opponent_goals = expected_total_goals - expected_team_goals
                
                # Calculate relative strengths
                attack_strength = max(0.1, expected_team_goals / 1.4)  # 1.4 is league average
                defense_strength = max(0.1, expected_opponent_goals / 1.4)
                
                attack_scores.append(attack_strength)
                defense_scores.append(defense_strength)
                form_scores.append(win_prob)
                
            except Exception as e:
                logger.warning(f"Error processing game data: {e}")
                continue
        
        if not attack_scores:
            return {'attack': 1.0, 'defense': 1.0, 'form': 0.5}
        
        # Calculate weighted averages
        weights = np.array(weights[:len(attack_scores)])
        weights = weights / weights.sum()
        
        avg_attack = np.average(attack_scores, weights=weights)
        avg_defense = np.average(defense_scores, weights=weights)
        avg_form = np.average(form_scores, weights=weights)
        
        return {
            'attack': float(avg_attack),
            'defense': float(avg_defense), 
            'form': float(avg_form)
        }
    
    def _find_optimal_lookback(self, team: str, odds_history: List[Dict]) -> int:
        """Find optimal lookback window for team strength calculation"""
        if len(odds_history) <= 5:
            return len(odds_history)
        
        # Test different window sizes
        window_sizes = [5, 8, 10, 12, 15]
        window_sizes = [w for w in window_sizes if w <= len(odds_history)]
        
        if not window_sizes:
            return min(5, len(odds_history))
        
        # Use consistency metric to choose optimal window
        best_window = window_sizes[0]
        best_consistency = float('inf')
        
        for window in window_sizes:
            recent_games = odds_history[-window:]
            strengths = self._calculate_strengths_from_odds(team, recent_games)
            
            # Calculate consistency (lower variance = better)
            if len(recent_games) >= 3:
                form_values = [game.get('form', 0.5) for game in recent_games[-3:]]
                consistency = np.var(form_values) if form_values else 1.0
                
                if consistency < best_consistency:
                    best_consistency = consistency
                    best_window = window
        
        return best_window
    
    def optimize_match_lambdas(self, odds_record: OddsRecord, 
                              initial_guess: Tuple[float, float] = (1.4, 1.1)) -> Tuple[float, float]:
        """
        Optimize lambda values (expected goals) to match betting odds
        
        Args:
            odds_record: Betting odds for the match
            initial_guess: Initial lambda values (home, away)
            
        Returns:
            Tuple of optimized (lambda_home, lambda_away)
        """
        try:
            # Convert odds to probabilities
            market_probs = remove_margin(
                odds_record.home_odds,
                odds_record.draw_odds,
                odds_record.away_odds
            )
            
            def objective(params):
                lambda_home, lambda_away = params
                if lambda_home <= 0 or lambda_away <= 0:
                    return 1e6
                
                # Calculate Poisson probabilities
                model_probs = self._match_probs_poisson(lambda_home, lambda_away)
                
                # Mean squared error
                mse = np.sum((np.array(model_probs) - np.array(market_probs))**2)
                return mse
            
            # Bounds for lambda values
            bounds = [(0.1, 5.0), (0.1, 5.0)]
            
            # Optimize using differential evolution (more robust)
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=100,
                popsize=10
            )
            
            if result.success:
                lambda_home, lambda_away = result.x
                return float(lambda_home), float(lambda_away)
            else:
                logger.warning(f"Optimization failed for match, using initial guess")
                return initial_guess
                
        except Exception as e:
            logger.error(f"Error optimizing match lambdas: {e}")
            return initial_guess
    
    def _match_probs_poisson(self, lambda_home: float, lambda_away: float) -> Tuple[float, float, float]:
        """Calculate match outcome probabilities using Poisson distribution"""
        try:
            # Create probability matrix
            probs = np.zeros((self.max_goals + 1, self.max_goals + 1))
            
            for i in range(self.max_goals + 1):
                for j in range(self.max_goals + 1):
                    probs[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            
            # Calculate outcome probabilities
            p_home = np.sum(np.tril(probs, -1))  # Home goals > Away goals
            p_draw = np.sum(np.diag(probs))      # Home goals = Away goals
            p_away = np.sum(np.triu(probs, 1))   # Away goals > Home goals
            
            return p_home, p_draw, p_away
            
        except Exception as e:
            logger.error(f"Error calculating Poisson probabilities: {e}")
            return 0.33, 0.33, 0.34
    
    def extract_team_strengths_from_fixtures(self, fixtures_with_odds: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Extract team strengths from a collection of fixtures with odds
        
        Args:
            fixtures_with_odds: List of fixtures with betting odds
            
        Returns:
            Dictionary mapping team names to their strength metrics
        """
        team_strengths = {}
        team_matches = {}
        
        # Group matches by team
        for fixture in fixtures_with_odds:
            home_team = fixture.get('home_team', '')
            away_team = fixture.get('away_team', '')
            
            if home_team not in team_matches:
                team_matches[home_team] = []
            if away_team not in team_matches:
                team_matches[away_team] = []
            
            # Add match data with home/away context
            home_data = fixture.copy()
            home_data['is_home'] = True
            team_matches[home_team].append(home_data)
            
            away_data = fixture.copy()
            away_data['is_home'] = False
            team_matches[away_team].append(away_data)
        
        # Calculate strengths for each team
        for team, matches in team_matches.items():
            if matches:
                strengths = self.calculate_team_strength(team, matches)
                team_strengths[team] = strengths
        
        return team_strengths
    
    def clear_cache(self):
        """Clear all cached data"""
        self._strength_cache.clear()
        self._match_cache.clear()
        logger.info("Cleared odds strength extractor cache")


class AdaptiveWindowSelector:
    """Dynamically select optimal window size for team strength calculation"""
    
    def __init__(self, min_window: int = 5, max_window: int = 15):
        self.min_window = min_window
        self.max_window = max_window
    
    def find_optimal_window(self, team: str, results_df: pd.DataFrame) -> int:
        """
        Find optimal window size based on team's match frequency and consistency
        
        Args:
            team: Team name
            results_df: Historical results dataframe
            
        Returns:
            Optimal window size
        """
        try:
            # Get team's matches
            team_matches = results_df[
                (results_df['HomeTeam'] == team) | 
                (results_df['AwayTeam'] == team)
            ].copy()
            
            if len(team_matches) < self.min_window:
                return len(team_matches)
            
            # Calculate match frequency (matches per month)
            if 'Date' in team_matches.columns:
                team_matches['Date'] = pd.to_datetime(team_matches['Date'])
                team_matches = team_matches.sort_values(by=['Date'])
                
                date_range = (team_matches['Date'].max() - team_matches['Date'].min()).days
                if date_range > 0:
                    match_frequency = len(team_matches) / (date_range / 30)  # matches per month
                else:
                    match_frequency = 1.0
            else:
                match_frequency = 1.0
            
            # Adjust window based on frequency
            if match_frequency > 4:  # High frequency (more than 4 matches/month)
                optimal_window = min(self.max_window, len(team_matches))
            elif match_frequency > 2:  # Medium frequency
                optimal_window = min(10, len(team_matches))
            else:  # Low frequency
                optimal_window = min(self.min_window + 2, len(team_matches))
            
            return max(self.min_window, optimal_window)
            
        except Exception as e:
            logger.error(f"Error finding optimal window for {team}: {e}")
            return self.min_window


class AdaptiveStrengthCalculator:
    """Combines odds-based and historical strength calculation with adaptive weighting"""
    
    def __init__(self):
        self.window_selector = AdaptiveWindowSelector()
        self.strength_extractor = OddsStrengthExtractor()
        
    def calculate_adaptive_strength(self, team: str, results_df: pd.DataFrame,
                                  odds_history: List[Dict]) -> Dict[str, float]:
        """
        Calculate team strength with adaptive window and combined methodology
        
        Args:
            team: Team name
            results_df: Historical results dataframe
            odds_history: List of odds records for the team
            
        Returns:
            Dictionary with comprehensive strength metrics
        """
        try:
            # Find optimal window
            optimal_window = self.window_selector.find_optimal_window(team, results_df)
            
            # Calculate strength with optimal window
            odds_strength = self.strength_extractor.calculate_team_strength(
                team, odds_history, lookback=optimal_window
            )
            
            # Add window and confidence metrics
            odds_strength['window_used'] = optimal_window
            odds_strength['confidence'] = min(1.0, len(odds_history) / 10)
            
            # Calculate data recency score
            if odds_history:
                latest_match = max(odds_history, key=lambda x: x.get('date', datetime.min))
                days_since_last = (datetime.now() - latest_match.get('date', datetime.now())).days
                odds_strength['recency_score'] = max(0.1, np.exp(-days_since_last / 30))
            else:
                odds_strength['recency_score'] = 0.1
            
            return odds_strength
            
        except Exception as e:
            logger.error(f"Error calculating adaptive strength for {team}: {e}")
            return {'attack': 1.0, 'defense': 1.0, 'form': 0.5, 'window_used': 5, 'confidence': 0.5}


def integrate_odds_strengths_with_historical(odds_strengths: Dict[str, Dict], 
                                           historical_strengths: pd.DataFrame,
                                           odds_weight: float = 0.3) -> pd.DataFrame:
    """
    Integrate odds-based strengths with historical strength calculations
    
    Args:
        odds_strengths: Dictionary of team strengths from odds
        historical_strengths: DataFrame with historical team strengths
        odds_weight: Weight to give to odds-based strengths (0.0 to 1.0)
        
    Returns:
        Enhanced DataFrame with integrated strengths
    """
    try:
        enhanced_strengths = historical_strengths.copy()
        
        # Add odds-based columns
        for team, strengths in odds_strengths.items():
            if team in enhanced_strengths.index:
                # Blend odds and historical strengths
                historical_attack = enhanced_strengths.loc[team, 'attack_strength']
                historical_defense = enhanced_strengths.loc[team, 'defense_strength']
                
                odds_attack = strengths.get('attack', 1.0)
                odds_defense = strengths.get('defense', 1.0)
                
                # Weighted combination
                blended_attack = (1 - odds_weight) * historical_attack + odds_weight * odds_attack
                blended_defense = (1 - odds_weight) * historical_defense + odds_weight * odds_defense
                
                # Update strengths
                enhanced_strengths.loc[team, 'attack_strength'] = blended_attack
                enhanced_strengths.loc[team, 'defense_strength'] = blended_defense
                
                # Add odds-specific metrics
                enhanced_strengths.loc[team, 'odds_attack'] = odds_attack
                enhanced_strengths.loc[team, 'odds_defense'] = odds_defense
                enhanced_strengths.loc[team, 'odds_form'] = strengths.get('form', 0.5)
                enhanced_strengths.loc[team, 'odds_confidence'] = strengths.get('confidence', 0.5)
        
        logger.info(f"Integrated odds strengths for {len(odds_strengths)} teams")
        return enhanced_strengths
        
    except Exception as e:
        logger.error(f"Error integrating odds strengths: {e}")
        return historical_strengths