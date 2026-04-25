from pydantic import BaseModel, validator
from datetime import datetime
from typing import Dict, Optional
import logging

def setup_logger(name):
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class OddsRecord(BaseModel):
    date: datetime
    home_team: str
    away_team: str
    home_odds: float
    draw_odds: float
    away_odds: float
    bookmaker: str = "aggregate"
    
    @validator('home_odds', 'draw_odds', 'away_odds')
    def odds_must_be_valid(cls, v):
        if v < 1.01 or v > 100:
            raise ValueError(f'Odds must be between 1.01 and 100, got {v}')
        return v

class OddsData:
    def __init__(self):
        self.odds_cache = {}  # {match_key: OddsRecord}
        self.logger = setup_logger(__name__)
    
    def add_match_odds(self, odds_record: OddsRecord):
        """Store validated odds with error handling"""
        try:
            key = f"{odds_record.date}_{odds_record.home_team}_{odds_record.away_team}"
            self.odds_cache[key] = odds_record
            self.logger.info(f"Added odds for {key}")
        except Exception as e:
            self.logger.error(f"Failed to add odds: {e}")
            raise
    
    def get_match_odds(self, home_team: str, away_team: str, date: datetime) -> Optional[OddsRecord]:
        """Retrieve odds for a specific match"""
        key = f"{date}_{home_team}_{away_team}"
        return self.odds_cache.get(key)
    
    def get_all_odds(self) -> Dict[str, OddsRecord]:
        """Get all cached odds"""
        return self.odds_cache.copy()
    
    def clear_cache(self):
        """Clear odds cache"""
        self.odds_cache.clear()
        self.logger.info("Odds cache cleared")