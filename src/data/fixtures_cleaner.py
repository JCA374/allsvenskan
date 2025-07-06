"""
Fixtures cleaner for upcoming_fixtures.csv to standardize team names
and format for Monte Carlo simulation
"""

import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class FixturesCleaner:
    def __init__(self):
        """Initialize fixtures cleaner with team name mapping"""
        # Mapping from upcoming_fixtures.csv team names to system team names
        self.team_name_mapping = {
            # Current system names from results.csv
            'GAIS': 'GAIS',
            'Malmo FF': 'Malmo FF',
            'Oster': 'Oster',
            'Mjallby': 'Mjallby',
            'Hammarby': 'Hammarby',
            'Varnamo': 'Varnamo',
            'Goteborg': 'Goteborg',
            'Sirius': 'Sirius',
            'Halmstad': 'Halmstad',
            'AIK': 'AIK',
            'Djurgarden': 'Djurgarden',
            'Degerfors': 'Degerfors',
            'Elfsborg': 'Elfsborg',
            'Hacken': 'Hacken',
            'Norrkoping': 'Norrkoping',
            'Brommapojkarna': 'Brommapojkarna',
            
            # Alternative name formats from upcoming_fixtures.csv
            'IFK Göteborg': 'Goteborg',
            'IFK Goteborg': 'Goteborg',
            'IFK Norrköping': 'Norrkoping',
            'IFK Norrkoping': 'Norrkoping',
            'IFK Värnamo': 'Varnamo',
            'IFK Varnamo': 'Varnamo',
            'Djurgårdens IF': 'Djurgarden',
            'Djurgardens IF': 'Djurgarden',
            'Malmö FF': 'Malmo FF',
            'Mjällby AIF': 'Mjallby',
            'Mjallby AIF': 'Mjallby',
            'Östers IF': 'Oster',
            'Osters IF': 'Oster',
            'BK Häcken': 'Hacken',
            'BK Hacken': 'Hacken',
            'IF Elfsborg': 'Elfsborg',
            'Halmstads BK': 'Halmstad',
            'IF Brommapojkarna': 'Brommapojkarna',
            'IK Sirius': 'Sirius',
            'Degerfors IF': 'Degerfors',
        }
    
    def normalize_team_name(self, team_name: str) -> str:
        """Normalize team name to match system convention"""
        if pd.isna(team_name) or team_name == '':
            return ''
            
        # Clean up the name
        team_name = str(team_name).strip()
        
        # Try exact match first
        if team_name in self.team_name_mapping:
            return self.team_name_mapping[team_name]
        
        # Try case-insensitive match
        for key, value in self.team_name_mapping.items():
            if key.lower() == team_name.lower():
                return value
        
        # Log unknown team names for manual review
        logger.warning(f"Unknown team name: {team_name}")
        return team_name
    
    def clean_fixtures_file(self, filepath: str) -> pd.DataFrame:
        """Clean the upcoming_fixtures.csv file and return standardized fixtures"""
        try:
            logger.info(f"Loading fixtures from {filepath}")
            
            # Read CSV with error handling for malformed lines
            try:
                df = pd.read_csv(filepath, on_bad_lines='skip')
            except Exception as e:
                logger.warning(f"Standard CSV read failed: {e}")
                # Try reading with different parameters
                df = pd.read_csv(filepath, error_bad_lines=False, warn_bad_lines=True)
            
            # Check if we have the expected columns
            if 'Home_Team' not in df.columns or 'Away_Team' not in df.columns:
                logger.error("Expected columns 'Home_Team' and 'Away_Team' not found")
                return pd.DataFrame()
            
            # Filter out completed matches (those with scores) and past matches
            if 'Status' in df.columns:
                # Remove matches that already have results (contain scores like "1-2")
                incomplete_matches = df[
                    (df['Status'].isna()) | 
                    (df['Status'] == '') | 
                    (~df['Status'].str.contains(r'\d+-\d+', na=False))
                ]
            else:
                incomplete_matches = df
            
            # Filter out matches that have already been played (past dates)
            from datetime import datetime
            today = datetime.now().date()
            
            # Convert Date column to datetime and filter for future matches only
            try:
                incomplete_matches['Date_parsed'] = pd.to_datetime(incomplete_matches['Date'], errors='coerce')
                future_matches = incomplete_matches[
                    incomplete_matches['Date_parsed'].dt.date >= today
                ].copy()
                logger.info(f"Filtered to {len(future_matches)} future matches from {len(incomplete_matches)} total")
            except Exception as e:
                logger.warning(f"Could not filter by date: {e}")
                future_matches = incomplete_matches
            
            # Create standardized fixtures dataframe
            fixtures = []
            
            for _, row in future_matches.iterrows():
                try:
                    # Parse date
                    date_str = row['Date']
                    
                    # Normalize team names - handle safely
                    try:
                        home_team_raw = row['Home_Team']
                        away_team_raw = row['Away_Team']
                        
                        home_team = self.normalize_team_name(str(home_team_raw))
                        away_team = self.normalize_team_name(str(away_team_raw))
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error extracting team names from row: {e}")
                        continue
                    
                    # Skip if we couldn't normalize team names
                    if not home_team or not away_team:
                        continue
                    
                    fixture = {
                        'Date': pd.to_datetime(date_str),
                        'HomeTeam': home_team,
                        'AwayTeam': away_team,
                        'Round': row.get('Round', ''),
                    }
                    
                    fixtures.append(fixture)
                    
                except Exception as e:
                    logger.warning(f"Error processing fixture row: {e}")
                    continue
            
            fixtures_df = pd.DataFrame(fixtures)
            
            if not fixtures_df.empty:
                # Sort by date
                fixtures_df = fixtures_df.sort_values('Date').reset_index(drop=True)
                logger.info(f"Successfully cleaned {len(fixtures_df)} fixtures")
            else:
                logger.warning("No valid fixtures found after cleaning")
            
            return fixtures_df
            
        except Exception as e:
            logger.error(f"Error cleaning fixtures file: {e}")
            return pd.DataFrame()
    
    def validate_team_names(self, fixtures_df: pd.DataFrame, results_df: Optional[pd.DataFrame] = None) -> Dict[str, list]:
        """Validate that fixture team names match existing team names from results"""
        validation_report = {
            'valid_teams': [],
            'unknown_teams': [],
            'fixture_teams': [],
            'results_teams': []
        }
        
        if fixtures_df.empty:
            return validation_report
        
        # Get unique teams from fixtures
        fixture_teams = set()
        fixture_teams.update(fixtures_df['HomeTeam'].unique())
        fixture_teams.update(fixtures_df['AwayTeam'].unique())
        fixture_teams = {team for team in fixture_teams if pd.notna(team) and team != ''}
        
        validation_report['fixture_teams'] = sorted(list(fixture_teams))
        
        if results_df is not None and not results_df.empty:
            # Get unique teams from results
            results_teams = set()
            results_teams.update(results_df['HomeTeam'].unique())
            results_teams.update(results_df['AwayTeam'].unique())
            results_teams = {team for team in results_teams if pd.notna(team) and team != ''}
            
            validation_report['results_teams'] = sorted(list(results_teams))
            
            # Check which fixture teams are valid
            validation_report['valid_teams'] = sorted(list(fixture_teams.intersection(results_teams)))
            validation_report['unknown_teams'] = sorted(list(fixture_teams - results_teams))
        else:
            validation_report['valid_teams'] = validation_report['fixture_teams']
        
        return validation_report
    
    def save_cleaned_fixtures(self, fixtures_df: pd.DataFrame, output_path: str = "data/clean/upcoming_fixtures_cleaned.csv"):
        """Save cleaned fixtures to CSV file"""
        try:
            if fixtures_df.empty:
                logger.warning("No fixtures to save")
                return False
            
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save with proper formatting
            fixtures_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(fixtures_df)} cleaned fixtures to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cleaned fixtures: {e}")
            return False