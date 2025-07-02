import pandas as pd
from datetime import datetime

class DataCleaner:
    def __init__(self):
        # API data is already clean and normalized
        pass
    
    def clean_data(self, df):
        """Clean API data and split into results and fixtures"""
        try:
            df_clean = df.copy()
            
            # API data is already well-formatted, just ensure proper data types
            # Split into results (completed matches) and fixtures (upcoming matches)
            results = df_clean.dropna(subset=['FTHG', 'FTAG']).copy()
            fixtures = df_clean[df_clean['FTHG'].isna() | df_clean['FTAG'].isna()].copy()
            
            # Clean up results dataframe - keep authentic team names from API
            if not results.empty:
                results = results[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
                results['FTHG'] = results['FTHG'].astype(int)
                results['FTAG'] = results['FTAG'].astype(int)
            
            # Clean up fixtures dataframe - keep authentic team names from API
            if not fixtures.empty:
                fixtures = fixtures[['Date', 'HomeTeam', 'AwayTeam']].copy()
            
            return results, fixtures
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def validate_data(self, results, fixtures):
        """Validate cleaned data"""
        try:
            # Check results
            if not results.empty:
                required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                if not all(col in results.columns for col in required_columns):
                    print("Missing required columns in results")
                    return False
                
                # Check for valid scores
                if (results['FTHG'] < 0).any() or (results['FTAG'] < 0).any():
                    print("Invalid negative scores found")
                    return False
            
            # Check fixtures
            if not fixtures.empty:
                required_columns = ['Date', 'HomeTeam', 'AwayTeam']
                if not all(col in fixtures.columns for col in required_columns):
                    print("Missing required columns in fixtures")
                    return False
            
            print(f"Data validation passed: {len(results)} results, {len(fixtures)} fixtures")
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False