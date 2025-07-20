import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.column_standardizer import ColumnStandardizer

def test_column_standardization():
    """Test that column standardization works correctly"""

    # Test various column name formats
    test_cases = [
        {'Home_Team': 'AIK', 'Away_Team': 'GAIS'},
        {'home_team': 'AIK', 'away_team': 'GAIS'},
        {'HomeTeam': 'AIK', 'AwayTeam': 'GAIS'},
    ]

    for test_df in test_cases:
        df = pd.DataFrame([test_df])
        standardized = ColumnStandardizer.standardize_columns(df)

        assert 'HomeTeam' in standardized.columns
        assert 'AwayTeam' in standardized.columns
        assert standardized['HomeTeam'].iloc[0] == 'AIK'
        assert standardized['AwayTeam'].iloc[0] == 'GAIS'

def test_merge_after_standardization():
    """Test that merges work after standardization"""

    # Create test fixtures with underscores
    fixtures = pd.DataFrame({
        'Home_Team': ['AIK', 'GAIS'],
        'Away_Team': ['Malmo FF', 'Hammarby'],
        'Date': ['2025-07-05', '2025-07-05']
    })

    # Create predictions with lowercase
    predictions = pd.DataFrame({
        'home_team': ['AIK', 'GAIS'],
        'away_team': ['Malmo FF', 'Hammarby'],
        'home_win': [0.45, 0.35]
    })

    # Standardize both
    fixtures_std = ColumnStandardizer.standardize_columns(fixtures)
    predictions_std = ColumnStandardizer.standardize_columns(predictions)

    # Merge should work now
    merged = predictions_std.merge(
        fixtures_std,
        left_on=['HomeTeam', 'AwayTeam'],
        right_on=['HomeTeam', 'AwayTeam'],
        how='left'
    )

    assert len(merged) == 2
    assert 'Date' in merged.columns

def test_july_5_fixtures():
    """Test that July 5th fixtures are properly processed"""
    
    # Test the specific July 5th fixtures mentioned in fix.md
    july_5_fixtures = pd.DataFrame({
        'Home_Team': ['GAIS', 'Oster', 'Hammarby'],
        'Away_Team': ['Malmo FF', 'Mjallby', 'Varnamo'],
        'Date': ['2025-07-05', '2025-07-05', '2025-07-05'],
        'Time': ['15:00', '15:00', '17:30']
    })
    
    # Standardize columns
    standardized = ColumnStandardizer.standardize_columns(july_5_fixtures)
    
    # Verify all rows are preserved
    assert len(standardized) == 3
    assert all(standardized['HomeTeam'] == ['GAIS', 'Oster', 'Hammarby'])
    assert all(standardized['AwayTeam'] == ['Malmo FF', 'Mjallby', 'Varnamo'])

if __name__ == "__main__":
    test_column_standardization()
    test_merge_after_standardization()
    test_july_5_fixtures()
    print("✅ All tests passed!")