import yaml
from functools import lru_cache
from typing import Dict, Any
import os

@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load configuration with caching"""
    config_path = os.path.join(os.path.dirname(__file__), 'odds_config.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default configuration if file not found
        return {
            'odds_integration': {
                'weights': {
                    'games_1_2': 0.7,
                    'games_3_5': 0.5,
                    'games_6_10': 0.3,
                    'games_11_plus': 0.1
                },
                'validation': {
                    'min_odds': 1.01,
                    'max_odds': 100.0,
                    'min_bookmakers': 1,
                    'max_margin_percent': 20.0
                },
                'api': {
                    'timeout_seconds': 30,
                    'retry_attempts': 3,
                    'retry_delay': 1.0
                },
                'cache': {
                    'ttl_seconds': 3600,
                    'max_size': 10000
                },
                'sports': {
                    'swedish_football': 'soccer_sweden_allsvenskan'
                },
                'regions': 'eu',
                'markets': 'h2h',
                'odds_format': 'decimal'
            }
        }

def get_odds_config() -> Dict[str, Any]:
    """Get odds integration configuration"""
    config = load_config()
    return config.get('odds_integration', {})

def get_api_config() -> Dict[str, Any]:
    """Get API configuration"""
    odds_config = get_odds_config()
    return odds_config.get('api', {})

def get_validation_config() -> Dict[str, Any]:
    """Get validation configuration"""
    odds_config = get_odds_config()
    return odds_config.get('validation', {})

def get_weights_config() -> Dict[str, Any]:
    """Get odds weighting configuration"""
    odds_config = get_odds_config()
    return odds_config.get('weights', {})