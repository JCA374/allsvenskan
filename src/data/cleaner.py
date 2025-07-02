import pandas as pd
import unidecode
from datetime import datetime
import re

class DataCleaner:
    def __init__(self):
        self.swedish_months = {
            'JANUARI': 'January', 'FEBRUARI': 'February', 'MARS': 'March',
            'APRIL': 'April', 'MAJ': 'May', 'JUNI': 'June',
            'JULI': 'July', 'AUGUSTI': 'August', 'SEPTEMBER': 'September',
            'OKTOBER': 'October', 'NOVEMBER': 'November', 'DECEMBER': 'December'
        }
        
        self.team_name_mapping = {
            'Malmö FF': 'Malmo FF',
            'IFK Göteborg': 'IFK Goteborg',
            'Djurgårdens IF': 'Djurgarden',
            'BK Häcken': 'Hacken',
            'Halmstads BK': 'Halmstad',
            'Mjällby AIF': 'Mjallby',
            'IK Sirius FK': 'Sirius',
            'Östersunds FK': 'Ostersund'
        }
    
    def clean_data(self, df):
        """Clean raw scraped data and split into results and fixtures"""
        try:
            df_clean = df.copy()
            
            # Normalize team names
            df_clean = self._normalize_team_names(df_clean)
            
            # Parse dates
            df_clean = self._parse_dates(df_clean)
            
            # Split match column into home and away teams
            df_clean = self._split_teams(df_clean)
            
            # Normalize goal columns
            df_clean = self._normalize_goals(df_clean)
            
            # Split into results (completed matches) and fixtures (upcoming matches)
            results = df_clean.dropna(subset=['FTHG', 'FTAG']).copy()
            fixtures = df_clean[df_clean['FTHG'].isna() | df_clean['FTAG'].isna()].copy()
            
            # Clean up results dataframe
            results = results[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
            results['FTHG'] = results['FTHG'].astype(int)
            results['FTAG'] = results['FTAG'].astype(int)
            
            # Clean up fixtures dataframe
            fixtures = fixtures[['Date', 'HomeTeam', 'AwayTeam']].copy()
            
            return results, fixtures
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _normalize_team_names(self, df):
        """Normalize team names by removing accents and standardizing"""
        if 'Match' in df.columns:
            # Apply unidecode to remove accents
            df['Match'] = df['Match'].apply(lambda x: unidecode.unidecode(str(x)) if pd.notna(x) else x)
            
            # Apply custom team name mappings
            for original, normalized in self.team_name_mapping.items():
                df['Match'] = df['Match'].str.replace(original, normalized)
        
        return df
    
    def _parse_dates(self, df):
        """Parse Swedish date formats"""
        if 'Date' not in df.columns:
            return df
            
        def parse_swedish_date(date_str):
            try:
                if pd.isna(date_str):
                    return date_str
                    
                date_str = str(date_str).upper().strip()
                
                # Replace Swedish day names (if present)
                day_names = {
                    'MÅNDAG': 'Monday', 'TISDAG': 'Tuesday', 'ONSDAG': 'Wednesday',
                    'TORSDAG': 'Thursday', 'FREDAG': 'Friday', 'LÖRDAG': 'Saturday', 'SÖNDAG': 'Sunday'
                }
                
                for swedish, english in day_names.items():
                    date_str = date_str.replace(swedish, english)
                
                # Replace Swedish month names
                for swedish, english in self.swedish_months.items():
                    date_str = date_str.replace(swedish, english)
                
                # Try to parse the date
                try:
                    # Format: "LÖRDAG 29 MARS" -> "Saturday 29 March"
                    return pd.to_datetime(date_str, format='%A %d %B')
                except:
                    try:
                        # Format: "29 MARS" -> "29 March"
                        return pd.to_datetime(date_str, format='%d %B')
                    except:
                        try:
                            # ISO format: "2024-07-01"
                            return pd.to_datetime(date_str)
                        except:
                            # Return original if all parsing fails
                            return date_str
                            
            except Exception as e:
                return date_str
        
        df['Date'] = df['Date'].apply(parse_swedish_date)
        return df
    
    def _split_teams(self, df):
        """Split the Match column into HomeTeam and AwayTeam"""
        if 'Match' not in df.columns:
            return df
            
        # Split on " - " separator
        teams = df['Match'].str.split(' - ', expand=True)
        
        if teams.shape[1] >= 2:
            df['HomeTeam'] = teams[0].str.strip()
            df['AwayTeam'] = teams[1].str.strip()
        else:
            # If split fails, try alternative separators
            teams = df['Match'].str.split(' vs ', expand=True)
            if teams.shape[1] >= 2:
                df['HomeTeam'] = teams[0].str.strip()
                df['AwayTeam'] = teams[1].str.strip()
            else:
                # Create dummy team names if parsing fails
                df['HomeTeam'] = 'Unknown Home'
                df['AwayTeam'] = 'Unknown Away'
        
        return df
    
    def _normalize_goals(self, df):
        """Normalize goal columns"""
        # Rename goal columns to standard format
        if 'HomeGoals' in df.columns:
            df.rename(columns={'HomeGoals': 'FTHG'}, inplace=True)
        if 'AwayGoals' in df.columns:
            df.rename(columns={'AwayGoals': 'FTAG'}, inplace=True)
        
        # Convert goals to numeric, handling None/NaN values
        if 'FTHG' in df.columns:
            df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
        if 'FTAG' in df.columns:
            df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
        
        return df
    
    def validate_data(self, results, fixtures):
        """Validate cleaned data"""
        validation_errors = []
        
        # Check results
        if len(results) > 0:
            if 'FTHG' not in results.columns or 'FTAG' not in results.columns:
                validation_errors.append("Results missing goal columns")
            
            if results['FTHG'].isna().any() or results['FTAG'].isna().any():
                validation_errors.append("Results contain missing goal data")
            
            if (results['FTHG'] < 0).any() or (results['FTAG'] < 0).any():
                validation_errors.append("Results contain negative goals")
        
        # Check fixtures
        if len(fixtures) > 0:
            required_columns = ['Date', 'HomeTeam', 'AwayTeam']
            for col in required_columns:
                if col not in fixtures.columns:
                    validation_errors.append(f"Fixtures missing {col} column")
        
        return validation_errors
