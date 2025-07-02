import requests
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AllsvenskanScraper:
    def __init__(self):
        # Primary data source: Football-Data CSV API
        self.csv_url = "https://www.football-data.co.uk/new/SWE.csv"
        # Backup API endpoint for fixtures
        self.api_url = "https://allsvenskan.se/wp-json/sef-leagues/v1/matches"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Connection': 'keep-alive'
        }
    
    def scrape_matches(self, years=None):
        """
        Scrape match data using Football-Data CSV API.
        Much simpler and more reliable than website scraping.
        """
        try:
            logger.info("Downloading Allsvenskan data from Football-Data CSV API...")
            
            # Download the CSV data
            response = requests.get(self.csv_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Load data into DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            logger.info(f"Downloaded {len(df)} total matches from Football-Data")
            
            # Parse dates (day-first format)
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            # Filter for the last 10 years if no specific years provided
            if years is None:
                current_year = datetime.now().year
                cutoff = current_year - 10
                df = df[df['Date'].dt.year >= cutoff].reset_index(drop=True)
                logger.info(f"Filtered to last 10 years ({cutoff}-{current_year}): {len(df)} matches")
            elif isinstance(years, list):
                df = df[df['Date'].dt.year.isin(years)].reset_index(drop=True)
                logger.info(f"Filtered to years {years}: {len(df)} matches")
            
            # Convert to our expected format
            df_formatted = self._format_football_data(df)
            
            logger.info(f"Successfully formatted {len(df_formatted)} matches")
            return df_formatted
            
        except requests.RequestException as e:
            logger.error(f"Failed to download from Football-Data API: {e}")
            return self._try_backup_api(years)
        except Exception as e:
            logger.error(f"Error processing Football-Data: {e}")
            return self._try_backup_api(years)
    
    def _format_football_data(self, df):
        """
        Convert Football-Data format to our expected format.
        Football-Data columns: Date, Home, Away, HG, AG, etc.
        Our format: Date, Match, HomeTeam, AwayTeam, FTHG, FTAG
        """
        formatted_matches = []
        
        for _, row in df.iterrows():
            try:
                # Create match string for compatibility with existing cleaner
                if pd.notna(row['HG']) and pd.notna(row['AG']):
                    # Completed match
                    match_str = f"{row['Home']} {int(row['HG'])}-{int(row['AG'])} {row['Away']}"
                else:
                    # Future fixture
                    match_str = f"{row['Home']} - {row['Away']}"
                
                formatted_match = {
                    'Date': row['Date'],
                    'Match': match_str,
                    'HomeTeam': row['Home'],
                    'AwayTeam': row['Away'],
                    'FTHG': row['HG'] if pd.notna(row['HG']) else None,
                    'FTAG': row['AG'] if pd.notna(row['AG']) else None
                }
                
                formatted_matches.append(formatted_match)
                
            except Exception as e:
                logger.warning(f"Error formatting match row: {e}")
                continue
        
        return pd.DataFrame(formatted_matches)
    
    def _try_backup_api(self, years):
        """
        Fallback to the allsvenskan.se API for fixtures if CSV fails.
        """
        try:
            logger.info("Trying backup API from allsvenskan.se...")
            
            # Get current year if years not specified
            if years is None:
                years = [datetime.now().year]
            elif not isinstance(years, list):
                years = [years]
            
            all_matches = []
            
            for year in years:
                params = {"league": "allsvenskan", "season": year}
                response = requests.get(self.api_url, params=params, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'matches' in data:
                        matches_df = pd.json_normalize(data['matches'])
                        if not matches_df.empty:
                            # Convert API format
                            matches_df['Date'] = pd.to_datetime(matches_df['kickoff'], utc=True)
                            # Add more API parsing logic here as needed
                            all_matches.append(matches_df)
                            logger.info(f"Got {len(matches_df)} matches from API for {year}")
            
            if all_matches:
                combined = pd.concat(all_matches, ignore_index=True)
                return self._format_api_data(combined)
            else:
                logger.warning("Backup API also failed, returning sample data")
                return self._create_sample_data()
                
        except Exception as e:
            logger.error(f"Backup API failed: {e}")
            return self._create_sample_data()
    
    def _format_api_data(self, df):
        """Format API data to our expected structure."""
        # This would need to be implemented based on the actual API response structure
        # For now, return empty DataFrame
        logger.warning("API data formatting not fully implemented")
        return pd.DataFrame()
    
    def _create_sample_data(self):
        """
        Create minimal sample data with real Swedish team names for testing.
        Only used when all data sources fail.
        """
        logger.warning("Creating sample data - all data sources failed")
        
        # Real Allsvenskan teams from Football-Data API
        teams = [
            'AIK', 'Brommapojkarna', 'Degerfors', 'Djurgarden', 'Elfsborg',
            'GAIS', 'Goteborg', 'Hacken', 'Halmstad', 'Hammarby',
            'Kalmar', 'Malmo FF', 'Mjallby', 'Norrkoping', 'Sirius',
            'Varnamo', 'Vasteras SK'
        ]
        
        sample_matches = []
        base_date = datetime(2025, 3, 1)  # Start from March 2025
        
        # Create a few sample completed matches
        for i in range(10):
            home = teams[i % len(teams)]
            away = teams[(i + 1) % len(teams)]
            home_goals = 1 + (i % 3)
            away_goals = i % 3
            
            match = {
                'Date': base_date.replace(day=base_date.day + i),
                'Match': f"{home} {home_goals}-{away_goals} {away}",
                'HomeTeam': home,
                'AwayTeam': away,
                'FTHG': home_goals,
                'FTAG': away_goals
            }
            sample_matches.append(match)
        
        # Add some future fixtures
        for i in range(10, 20):
            home = teams[i % len(teams)]
            away = teams[(i + 2) % len(teams)]
            
            match = {
                'Date': base_date.replace(day=base_date.day + i),
                'Match': f"{home} - {away}",
                'HomeTeam': home,
                'AwayTeam': away,
                'FTHG': None,
                'FTAG': None
            }
            sample_matches.append(match)
        
        return pd.DataFrame(sample_matches)

    def validate_match_data(self, match_data):
        """Validate that we have reasonable match data."""
        if match_data.empty:
            return False
        
        required_columns = ['Date', 'Match', 'HomeTeam', 'AwayTeam']
        missing_columns = [col for col in required_columns if col not in match_data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        if len(match_data) < 5:
            logger.warning(f"Very few matches found: {len(match_data)}")
            return False
        
        logger.info(f"Validation passed: {len(match_data)} matches with required columns")
        return True