"""
Demonstration utility for enhanced team strength calculations
Shows how the odds-based strength extraction improves over historical-only methods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

from src.data.strength import TeamStrengthCalculator
from src.data.odds_schema import OddsRecord

logger = logging.getLogger(__name__)

def create_sample_odds_data() -> List[Dict]:
    """Create sample odds data for demonstration"""
    teams = ['Hammarby', 'AIK', 'Djurgarden', 'Malmo FF', 'Goteborg']
    sample_odds = []
    
    # Create sample matches with realistic odds
    base_date = datetime.now()
    for i in range(10):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        # Create realistic odds based on team "strength"
        home_strength = np.random.uniform(0.8, 1.5)
        away_strength = np.random.uniform(0.8, 1.5)
        
        # Convert to betting odds (inverse relationship)
        home_odds = 1.5 + (2.0 / home_strength)
        away_odds = 1.5 + (2.0 / away_strength)
        draw_odds = 3.2 + np.random.uniform(-0.5, 0.5)
        
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_odds': home_odds,
            'draw_odds': draw_odds,
            'away_odds': away_odds,
            'date': base_date + timedelta(days=i),
            'bookmaker': 'demo',
            'is_home': True  # For strength calculation
        }
        
        sample_odds.append(match_data)
    
    return sample_odds

def demonstrate_enhanced_calculations(results_df: pd.DataFrame) -> Dict:
    """
    Demonstrate the difference between historical and enhanced calculations
    
    Args:
        results_df: Historical match results
        
    Returns:
        Dictionary with comparison results
    """
    try:
        logger.info("Starting enhanced strength calculation demonstration")
        
        # Historical-only calculation
        historical_calc = TeamStrengthCalculator(use_odds_integration=False)
        historical_strengths = historical_calc.calculate_strengths(results_df)
        
        # Enhanced calculation with sample odds
        sample_odds = create_sample_odds_data()
        odds_data = {'fixtures': sample_odds}
        
        enhanced_calc = TeamStrengthCalculator(use_odds_integration=True)
        enhanced_strengths = enhanced_calc.calculate_strengths(results_df, odds_data)
        
        # Compare results
        comparison = {}
        
        if not historical_strengths.empty and not enhanced_strengths.empty:
            common_teams = set(historical_strengths.index) & set(enhanced_strengths.index)
            
            for team in list(common_teams)[:5]:  # Show top 5 teams
                comparison[team] = {
                    'historical_attack': historical_strengths.loc[team, 'attack_strength'],
                    'historical_defense': historical_strengths.loc[team, 'defense_strength'],
                    'enhanced_attack': enhanced_strengths.loc[team, 'attack_strength'],
                    'enhanced_defense': enhanced_strengths.loc[team, 'defense_strength'],
                }
                
                # Add odds-specific metrics if available
                if 'odds_attack' in enhanced_strengths.columns:
                    comparison[team]['odds_attack'] = enhanced_strengths.loc[team, 'odds_attack']
                    comparison[team]['odds_form'] = enhanced_strengths.loc[team, 'odds_form']
        
        logger.info(f"Demonstration completed for {len(comparison)} teams")
        return {
            'comparison': comparison,
            'historical_count': len(historical_strengths),
            'enhanced_count': len(enhanced_strengths),
            'sample_odds_count': len(sample_odds)
        }
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        return {'error': str(e)}

def create_odds_extraction_example():
    """Create an example showing odds-to-lambda extraction"""
    try:
        # Sample odds record
        odds_record = OddsRecord(
            date=datetime.now(),
            home_team="Hammarby",
            away_team="AIK",
            home_odds=2.1,
            draw_odds=3.3,
            away_odds=3.8,
            bookmaker="demo"
        )
        
        # Initialize calculator
        calc = TeamStrengthCalculator(use_odds_integration=True)
        
        # Extract lambda values
        lambda_home, lambda_away = calc.extract_lambda_values_from_odds(odds_record)
        
        return {
            'match': f"{odds_record.home_team} vs {odds_record.away_team}",
            'odds': [odds_record.home_odds, odds_record.draw_odds, odds_record.away_odds],
            'extracted_goals': [lambda_home, lambda_away],
            'implied_total_goals': lambda_home + lambda_away
        }
        
    except Exception as e:
        logger.error(f"Error creating odds extraction example: {e}")
        return {'error': str(e)}

def validate_implementation() -> Dict:
    """Validate that the implementation is working correctly"""
    validation_results = {
        'odds_extractor_available': False,
        'adaptive_calculator_available': False,
        'integration_function_available': False,
        'team_calculator_enhanced': False
    }
    
    try:
        # Test odds strength extractor
        from src.data.odds_strength_extractor import OddsStrengthExtractor
        extractor = OddsStrengthExtractor()
        validation_results['odds_extractor_available'] = True
        
        # Test adaptive calculator
        from src.data.odds_strength_extractor import AdaptiveStrengthCalculator
        adaptive_calc = AdaptiveStrengthCalculator()
        validation_results['adaptive_calculator_available'] = True
        
        # Test integration function
        from src.data.odds_strength_extractor import integrate_odds_strengths_with_historical
        validation_results['integration_function_available'] = True
        
        # Test enhanced team calculator
        calc = TeamStrengthCalculator(use_odds_integration=True)
        validation_results['team_calculator_enhanced'] = calc.use_odds_integration
        
    except Exception as e:
        logger.warning(f"Validation error: {e}")
        validation_results['error'] = str(e)
    
    return validation_results

if __name__ == "__main__":
    # Run validation
    validation = validate_implementation()
    print("Implementation Validation:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Create sample data for demonstration
    sample_results = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=20, freq='W'),
        'HomeTeam': np.random.choice(['Hammarby', 'AIK', 'Djurgarden'], 20),
        'AwayTeam': np.random.choice(['Malmo FF', 'Goteborg', 'Elfsborg'], 20),
        'FTHG': np.random.poisson(1.4, 20),
        'FTAG': np.random.poisson(1.1, 20)
    })
    
    # Run demonstration
    demo_results = demonstrate_enhanced_calculations(sample_results)
    print("\nDemonstration Results:")
    print(f"Historical teams: {demo_results.get('historical_count', 0)}")
    print(f"Enhanced teams: {demo_results.get('enhanced_count', 0)}")
    print(f"Sample odds: {demo_results.get('sample_odds_count', 0)}")