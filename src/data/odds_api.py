import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import logging
import os
from src.config.config_loader import get_odds_config, get_api_config
from src.data.odds_schema import OddsRecord
from src.utils.odds_converter import validate_odds, find_best_odds

logger = logging.getLogger(__name__)

class OddsAPI:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("ODDS_API_KEY must be provided or set in environment variables")
        
        self.config = get_odds_config()
        self.api_config = get_api_config()
        self.base_url = "https://api.the-odds-api.com/v4"
        
        # Configuration
        self.regions = self.config.get('regions', 'eu')
        self.markets = self.config.get('markets', 'h2h')
        self.odds_format = self.config.get('odds_format', 'decimal')
        self.sport = self.config.get('sports', {}).get('swedish_football', 'soccer_sweden_allsvenskan')
        
        # API settings
        self.timeout = self.api_config.get('timeout_seconds', 30)
        self.retry_attempts = self.api_config.get('retry_attempts', 3)
        self.retry_delay = self.api_config.get('retry_delay', 1.0)
        
        self.logger = logging.getLogger(__name__)
    
    def get_upcoming_matches_odds(self) -> List[OddsRecord]:
        """Fetch odds for upcoming matches"""
        url = f"{self.base_url}/sports/{self.sport}/odds"
        params = {
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": self.odds_format,
            "apiKey": self.api_key
        }
        
        try:
            response = self._make_request(url, params)
            if response and response.status_code == 200:
                data = response.json()
                return self._parse_odds_response(data)
            else:
                self.logger.error(f"API request failed: {response.status_code if response else 'No response'}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching odds: {e}")
            return []
    
    def get_match_odds(self, home_team: str, away_team: str, date: Optional[datetime] = None) -> Optional[OddsRecord]:
        """Get odds for a specific match"""
        all_odds = self.get_upcoming_matches_odds()
        
        for odds_record in all_odds:
            if (odds_record.home_team.lower() == home_team.lower() and 
                odds_record.away_team.lower() == away_team.lower()):
                # If date is specified, check if it matches (within 24 hours)
                if date:
                    time_diff = abs((odds_record.date - date).total_seconds())
                    if time_diff > 86400:  # 24 hours
                        continue
                return odds_record
        
        return None
    
    def _make_request(self, url: str, params: Dict) -> Optional[requests.Response]:
        """Make API request with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                self.logger.info(f"Making API request (attempt {attempt + 1}/{self.retry_attempts})")
                response = requests.get(url, params=params, timeout=self.timeout)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    self.logger.warning(f"API returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error on attempt {attempt + 1}: {e}")
            
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay)
        
        return None
    
    def _parse_odds_response(self, data: List[Dict]) -> List[OddsRecord]:
        """Parse API response into OddsRecord objects"""
        odds_records = []
        
        for match in data:
            try:
                home_team = match["home_team"]
                away_team = match["away_team"]
                commence_time = datetime.fromisoformat(match["commence_time"].replace('Z', '+00:00'))
                bookmakers = match.get("bookmakers", [])
                
                if not bookmakers:
                    self.logger.warning(f"No bookmakers found for {home_team} vs {away_team}")
                    continue
                
                # Collect odds from all bookmakers
                all_odds = []
                for bookmaker in bookmakers:
                    bm_name = bookmaker.get("title", "unknown")
                    markets = bookmaker.get("markets", [])
                    
                    if not markets:
                        continue
                    
                    outcomes = markets[0].get("outcomes", [])
                    if len(outcomes) != 3:  # Should have home, draw, away
                        continue
                    
                    # Parse outcomes
                    odds_dict = {}
                    for outcome in outcomes:
                        name = outcome["name"]
                        price = outcome["price"]
                        
                        if name == home_team:
                            odds_dict['home_odds'] = price
                        elif name == away_team:
                            odds_dict['away_odds'] = price
                        elif name == "Draw":
                            odds_dict['draw_odds'] = price
                    
                    # Only add if we have all three odds
                    if len(odds_dict) == 3:
                        odds_dict['bookmaker'] = bm_name
                        all_odds.append(odds_dict)
                
                if all_odds:
                    # Use best odds across bookmakers
                    best_home, best_draw, best_away = find_best_odds(all_odds)
                    
                    # Validate odds
                    if validate_odds(best_home, best_draw, best_away):
                        odds_record = OddsRecord(
                            date=commence_time,
                            home_team=home_team,
                            away_team=away_team,
                            home_odds=best_home,
                            draw_odds=best_draw,
                            away_odds=best_away,
                            bookmaker="best_odds"
                        )
                        odds_records.append(odds_record)
                        self.logger.info(f"Added odds for {home_team} vs {away_team}")
                    else:
                        self.logger.warning(f"Invalid odds for {home_team} vs {away_team}")
                
            except Exception as e:
                self.logger.error(f"Error parsing match data: {e}")
                continue
        
        self.logger.info(f"Successfully parsed {len(odds_records)} matches with odds")
        return odds_records
    
    def check_api_usage(self) -> Dict[str, int]:
        """Check API usage statistics"""
        url = f"{self.base_url}/sports/{self.sport}/odds"
        params = {
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": self.odds_format,
            "apiKey": self.api_key
        }
        
        try:
            response = self._make_request(url, params)
            if response:
                usage_info = {
                    'requests_used': int(response.headers.get('x-requests-used', 0)),
                    'requests_remaining': int(response.headers.get('x-requests-remaining', 0))
                }
                self.logger.info(f"API Usage: {usage_info}")
                return usage_info
        except Exception as e:
            self.logger.error(f"Error checking API usage: {e}")
        
        return {'requests_used': 0, 'requests_remaining': 0}
    
    def normalize_team_name(self, team_name: str) -> str:
        """Normalize team names to match our system"""
        # Common mappings for Swedish teams
        name_mappings = {
            "IFK Göteborg": "Goteborg",
            "IFK Goteborg": "Goteborg", 
            "Göteborg": "Goteborg",
            "Malmö FF": "Malmo FF",
            "Malmo": "Malmo FF",
            "AIK Stockholm": "AIK",
            "Djurgårdens IF": "Djurgarden",
            "Djurgarden IF": "Djurgarden",
            "Hammarby IF": "Hammarby",
            "IF Elfsborg": "Elfsborg",
            "Real Sociedad": "Real Sociedad"  # Keep as is
        }
        
        # Try direct mapping first
        if team_name in name_mappings:
            return name_mappings[team_name]
        
        # Basic normalization
        normalized = team_name.strip()
        normalized = normalized.replace("IFK ", "").replace("IF ", "").replace(" IF", "")
        normalized = normalized.replace("ö", "o").replace("å", "a").replace("ä", "a")
        
        return normalized