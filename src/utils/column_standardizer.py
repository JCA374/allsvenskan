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
        'ftag': 'FTAG'
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