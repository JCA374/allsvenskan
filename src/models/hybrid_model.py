import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from datetime import datetime
import pandas as pd

from src.models.poisson_model import PoissonModel
from src.data.odds_schema import OddsRecord
from src.utils.odds_converter import remove_margin, calculate_implied_probabilities
from src.config.config_loader import get_weights_config

logger = logging.getLogger(__name__)

class HybridPoissonOddsModel:
    """
    Hybrid model that combines Poisson-based predictions with betting odds
    """
    
    def __init__(self, poisson_model: PoissonModel):
        self.poisson_model = poisson_model
        self.weights_config = get_weights_config()
        self.logger = logging.getLogger(__name__)
        
    def predict_with_odds(self, home_team: str, away_team: str, 
                         odds_record: Optional[OddsRecord] = None,
                         season_progress: float = 0.5) -> Tuple[float, float, float]:
        """
        Predict match outcome combining Poisson model with betting odds
        
        Args:
            home_team: Name of home team
            away_team: Name of away team
            odds_record: Betting odds for the match (optional)
            season_progress: Progress through season (0.0 = start, 1.0 = end)
            
        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob)
        """
        
        # Get Poisson prediction
        poisson_result = self.poisson_model.predict_outcome_probabilities(home_team, away_team)
        poisson_probs = (poisson_result['home_win'], poisson_result['draw'], poisson_result['away_win'])
        
        if odds_record is None:
            # No odds available, return pure Poisson prediction
            self.logger.info(f"No odds available for {home_team} vs {away_team}, using pure Poisson")
            return poisson_probs
        
        # Get odds-implied probabilities
        try:
            odds_probs = remove_margin(
                odds_record.home_odds,
                odds_record.draw_odds, 
                odds_record.away_odds
            )
        except Exception as e:
            self.logger.error(f"Error processing odds: {e}")
            return poisson_probs
        
        # Calculate odds weight based on season progress
        odds_weight = self._calculate_odds_weight(season_progress)
        poisson_weight = 1.0 - odds_weight
        
        # Combine probabilities
        combined_probs = (
            poisson_weight * poisson_probs[0] + odds_weight * odds_probs[0],
            poisson_weight * poisson_probs[1] + odds_weight * odds_probs[1], 
            poisson_weight * poisson_probs[2] + odds_weight * odds_probs[2]
        )
        
        # Ensure probabilities sum to 1
        total = sum(combined_probs)
        combined_probs = (combined_probs[0] / total, combined_probs[1] / total, combined_probs[2] / total)
        
        self.logger.info(f"Combined prediction for {home_team} vs {away_team}: "
                        f"Poisson={poisson_probs}, Odds={odds_probs}, "
                        f"Combined={combined_probs} (odds_weight={odds_weight:.2f})")
        
        return combined_probs
    
    def _calculate_odds_weight(self, season_progress: float) -> float:
        """
        Calculate how much weight to give to odds vs Poisson model
        Early season: Higher odds weight (less historical data)
        Late season: Lower odds weight (more historical data available)
        """
        
        # Convert season progress to number of games played estimate
        # Assuming ~30 games per season, this is rough estimate
        estimated_games = int(season_progress * 30)
        
        # Get weight based on games played
        if estimated_games <= 2:
            weight = self.weights_config.get('games_1_2', 0.7)
        elif estimated_games <= 5:
            weight = self.weights_config.get('games_3_5', 0.5)
        elif estimated_games <= 10:
            weight = self.weights_config.get('games_6_10', 0.3)
        else:
            weight = self.weights_config.get('games_11_plus', 0.1)
        
        self.logger.debug(f"Season progress: {season_progress:.2f}, "
                         f"estimated games: {estimated_games}, odds weight: {weight}")
        
        return weight
    
    def predict_match_detailed(self, home_team: str, away_team: str,
                              odds_record: Optional[OddsRecord] = None,
                              season_progress: float = 0.5) -> Dict:
        """
        Get detailed prediction including component breakdowns
        """
        
        # Get individual predictions
        poisson_result = self.poisson_model.predict_outcome_probabilities(home_team, away_team)
        poisson_probs = (poisson_result['home_win'], poisson_result['draw'], poisson_result['away_win'])
        
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'poisson_probs': poisson_probs,
            'season_progress': season_progress
        }
        
        if odds_record:
            try:
                odds_probs = remove_margin(
                    odds_record.home_odds,
                    odds_record.draw_odds,
                    odds_record.away_odds
                )
                
                odds_weight = self._calculate_odds_weight(season_progress)
                combined_probs = self.predict_with_odds(
                    home_team, away_team, odds_record, season_progress
                )
                
                result.update({
                    'odds_record': odds_record,
                    'odds_probs': odds_probs,
                    'odds_weight': odds_weight,
                    'combined_probs': combined_probs,
                    'has_odds': True
                })
                
            except Exception as e:
                self.logger.error(f"Error processing odds in detailed prediction: {e}")
                result.update({
                    'combined_probs': poisson_probs,
                    'has_odds': False,
                    'error': str(e)
                })
        else:
            result.update({
                'combined_probs': poisson_probs,
                'has_odds': False
            })
        
        return result
    
    def get_expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Get expected goals from Poisson model"""
        return self.poisson_model.predict_match(home_team, away_team)
    
    def calculate_value_bets(self, prediction_probs: Tuple[float, float, float],
                           odds_record: OddsRecord, min_edge: float = 0.05) -> List[Dict]:
        """
        Identify value betting opportunities
        
        Args:
            prediction_probs: Our model's probabilities
            odds_record: Market odds
            min_edge: Minimum edge required to consider a bet
            
        Returns:
            List of value bet opportunities
        """
        
        value_bets = []
        outcomes = ['home', 'draw', 'away']
        market_odds = [odds_record.home_odds, odds_record.draw_odds, odds_record.away_odds]
        
        for i, (outcome, prob, odds) in enumerate(zip(outcomes, prediction_probs, market_odds)):
            # Calculate expected value
            expected_return = prob * odds
            edge = expected_return - 1.0
            
            if edge > min_edge:
                value_bets.append({
                    'outcome': outcome,
                    'our_probability': prob,
                    'market_odds': odds,
                    'implied_probability': 1/odds,
                    'expected_return': expected_return,
                    'edge': edge,
                    'edge_percent': edge * 100
                })
        
        # Sort by edge descending
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        
        if value_bets:
            self.logger.info(f"Found {len(value_bets)} value bets for {odds_record.home_team} vs {odds_record.away_team}")
        
        return value_bets
    
    def evaluate_prediction_accuracy(self, predictions: List[Dict], 
                                   actual_results: List[Dict]) -> Dict:
        """
        Evaluate model accuracy against actual results
        
        Args:
            predictions: List of prediction dictionaries
            actual_results: List of actual match results
            
        Returns:
            Dictionary with accuracy metrics
        """
        
        if len(predictions) != len(actual_results):
            raise ValueError("Predictions and results must have same length")
        
        correct_predictions = 0
        total_log_loss = 0
        total_brier_score = 0
        
        for pred, actual in zip(predictions, actual_results):
            pred_probs = pred.get('combined_probs', pred.get('poisson_probs'))
            actual_outcome = actual.get('outcome')  # 0=home, 1=draw, 2=away
            
            # Accuracy
            predicted_outcome = np.argmax(pred_probs)
            if predicted_outcome == actual_outcome:
                correct_predictions += 1
            
            # Log loss
            actual_prob = pred_probs[actual_outcome]
            log_loss = -np.log(max(actual_prob, 1e-15))
            total_log_loss += log_loss
            
            # Brier score
            outcome_vector = [0, 0, 0]
            outcome_vector[actual_outcome] = 1
            brier = sum((pred_probs[i] - outcome_vector[i])**2 for i in range(3))
            total_brier_score += brier
        
        n = len(predictions)
        metrics = {
            'accuracy': correct_predictions / n,
            'log_loss': total_log_loss / n,
            'brier_score': total_brier_score / n,
            'total_predictions': n
        }
        
        self.logger.info(f"Model evaluation: Accuracy={metrics['accuracy']:.1%}, "
                        f"Log Loss={metrics['log_loss']:.3f}, "
                        f"Brier Score={metrics['brier_score']:.3f}")
        
        return metrics