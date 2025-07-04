import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

def odds_to_probability(decimal_odds: float) -> float:
    """Convert single odds to probability with validation"""
    if decimal_odds < 1.01:
        logger.warning(f"Odds {decimal_odds} below minimum, setting to 1.01")
        decimal_odds = 1.01
    return 1.0 / decimal_odds

def remove_margin(home_odds: float, draw_odds: float, away_odds: float) -> Tuple[float, float, float]:
    """Remove bookmaker margin with error handling"""
    try:
        total = 1/home_odds + 1/draw_odds + 1/away_odds
        if total < 1.0:
            logger.warning(f"Negative margin detected: {total}, possible arbitrage")
        
        probs = (1/home_odds/total, 1/draw_odds/total, 1/away_odds/total)
        
        # Sanity check
        if not (0.99 < sum(probs) < 1.01):
            logger.error(f"Probability sum {sum(probs)} not close to 1.0")
        
        return probs
    except ZeroDivisionError:
        logger.error("Zero odds detected")
        return (0.33, 0.33, 0.34)  # Fallback to uniform

def calculate_implied_probabilities(home_odds: float, draw_odds: float, away_odds: float) -> Tuple[float, float, float]:
    """Calculate implied probabilities from decimal odds"""
    home_prob = odds_to_probability(home_odds)
    draw_prob = odds_to_probability(draw_odds)
    away_prob = odds_to_probability(away_odds)
    
    return (home_prob, draw_prob, away_prob)

def calculate_margin(home_odds: float, draw_odds: float, away_odds: float) -> float:
    """Calculate bookmaker margin percentage"""
    total_implied = 1/home_odds + 1/draw_odds + 1/away_odds
    margin = (total_implied - 1) * 100
    return margin

def calculate_fair_odds(home_odds: float, draw_odds: float, away_odds: float) -> Tuple[float, float, float]:
    """Calculate fair odds after removing margin"""
    probs = remove_margin(home_odds, draw_odds, away_odds)
    fair_odds = (1/probs[0], 1/probs[1], 1/probs[2])
    return fair_odds

def validate_odds(home_odds: float, draw_odds: float, away_odds: float) -> bool:
    """Validate that odds are reasonable"""
    try:
        # Check basic range
        for odds in [home_odds, draw_odds, away_odds]:
            if odds < 1.01 or odds > 100:
                logger.error(f"Odds {odds} outside valid range [1.01, 100]")
                return False
        
        # Check margin is reasonable (should be positive and under 20%)
        margin = calculate_margin(home_odds, draw_odds, away_odds)
        if margin < -5 or margin > 20:
            logger.warning(f"Unusual margin: {margin:.2f}%")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating odds: {e}")
        return False

def find_best_odds(odds_list: list) -> Tuple[float, float, float]:
    """Find best odds from multiple bookmakers"""
    if not odds_list:
        return (2.0, 3.0, 3.5)  # Default odds
    
    # Extract odds for each outcome
    home_odds = [odds.get('home_odds', 1.01) for odds in odds_list if odds.get('home_odds')]
    draw_odds = [odds.get('draw_odds', 1.01) for odds in odds_list if odds.get('draw_odds')]
    away_odds = [odds.get('away_odds', 1.01) for odds in odds_list if odds.get('away_odds')]
    
    # Take best (highest) odds for each outcome
    best_home = max(home_odds) if home_odds else 2.0
    best_draw = max(draw_odds) if draw_odds else 3.0
    best_away = max(away_odds) if away_odds else 3.5
    
    return (best_home, best_draw, best_away)