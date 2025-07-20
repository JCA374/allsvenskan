"""
Column name standardization utility for consistent data processing
Ensures all dataframes use the same column naming convention
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ColumnStandardizer:
    """Ensures consistent column naming across all dataframes"""

    # Define the canonical column names
    STANDARD_COLUMNS = {
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'Home_Team': 'HomeTeam',
        'Away_Team': 'AwayTeam',
        'hometeam': 'HomeTeam',
        'awayteam': 'AwayTeam',
        'date': 'Date',
        'DATE': 'Date',
        'fthg': 'FTHG',
        'ftag': 'FTAG',
        'FTHG': 'FTHG',
        'FTAG': 'FTAG',
        'home_goals': 'FTHG',
        'away_goals': 'FTAG',
        'Home_Goals': 'FTHG',
        'Away_Goals': 'FTAG'
    }

    @classmethod
    def standardize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names in a dataframe"""
        df = df.copy()

        # Create mapping for existing columns
        rename_map = {}
        for col in df.columns:
            # Check exact match first
            if col in cls.STANDARD_COLUMNS:
                rename_map[col] = cls.STANDARD_COLUMNS[col]
            # Check lowercase match
            elif col.lower() in cls.STANDARD_COLUMNS:
                rename_map[col] = cls.STANDARD_COLUMNS[col.lower()]

        # Apply renaming
        if rename_map:
            logger.info(f"Standardizing columns: {rename_map}")
            df = df.rename(columns=rename_map)

        return df

    @classmethod
    def validate_required_columns(cls, df: pd.DataFrame, required: list) -> bool:
        """Validate that dataframe has required columns"""
        missing = set(required) - set(df.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        return True

    @classmethod
    def standardize_team_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Specifically standardize team name columns"""
        df = df.copy()
        
        # Handle various team column formats
        team_column_mapping = {
            'Home_Team': 'HomeTeam',
            'Away_Team': 'AwayTeam', 
            'home_team': 'HomeTeam',
            'away_team': 'AwayTeam',
            'hometeam': 'HomeTeam',
            'awayteam': 'AwayTeam'
        }
        
        # Apply team column standardization
        for old_col, new_col in team_column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                logger.info(f"Renamed {old_col} to {new_col}")
        
        return df

    @classmethod
    def get_column_mapping(cls, df: pd.DataFrame) -> dict:
        """Get the column mapping that would be applied to a dataframe"""
        mapping = {}
        for col in df.columns:
            if col in cls.STANDARD_COLUMNS:
                mapping[col] = cls.STANDARD_COLUMNS[col]
            elif col.lower() in cls.STANDARD_COLUMNS:
                mapping[col] = cls.STANDARD_COLUMNS[col.lower()]
        return mapping