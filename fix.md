# Monte Carlo Simulation Debug Fix

## Issue Identified

The Monte Carlo simulation is not getting the upcoming games correctly because of several interconnected issues:

### 1. **File Path Mismatch**
- Your code references: `data/cleaned/upcoming_fixtures.csv`
- But the actual file is at: `data/clean/upcoming_fixtures.csv`

### 2. **Column Name Inconsistency**
- The upcoming_fixtures.csv file uses columns: `Home_Team`, `Away_Team`
- But the fixture predictions processing expects: `HomeTeam`, `AwayTeam`

### 3. **Data Structure Problem**
The current CSV has this structure:
```csv
Round,Date,Day,Time,Home_Team,Away_Team,
19,2025-07-05,LÖRDAG,15:00,GAIS,Malmo FF,
19,2025-07-05,LÖRDAG,15:00,Oster,Mjallby,
19,2025-07-05,LÖRDAG,17:30,Hammarby,Varnamo,
```

But the code expects:
```csv
Date,HomeTeam,AwayTeam
2025-07-05,GAIS,Malmo FF
2025-07-05,Oster,Mjallby
2025-07-05,Hammarby,Varnamo
```

## Solution

### Quick Fix 1: Update File Path in app.py

Replace this line in `app.py`:
```python
upcoming_fixtures_path = "data/clean/upcoming_fixtures.csv"
```

With:
```python
upcoming_fixtures_path = "data/cleaned/upcoming_fixtures.csv"
```

### Quick Fix 2: Update the MonteCarloSimulator Class

In `src/simulation/simulator.py`, update the `_load_upcoming_fixtures_directly` method to handle the correct column names:

```python
@staticmethod
def _load_upcoming_fixtures_directly(filepath):
    """Load upcoming fixtures directly from CSV with robust parsing"""
    try:
        # Read CSV with error handling
        df = pd.read_csv(filepath, on_bad_lines='skip')
        
        # Handle different column name formats
        if 'Home_Team' in df.columns and 'Away_Team' in df.columns:
            # Rename columns to match expected format
            df = df.rename(columns={
                'Home_Team': 'HomeTeam',
                'Away_Team': 'AwayTeam'
            })
        
        # Ensure required columns exist
        if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
            raise ValueError("CSV must contain HomeTeam and AwayTeam columns")
        
        # Clean and standardize team names
        team_mapping = {
            'GAIS': 'GAIS', 'Malmo FF': 'Malmo FF', 'Malmö FF': 'Malmo FF',
            'Oster': 'Oster', 'Östers IF': 'Oster', 'Osters IF': 'Oster',
            'Mjallby': 'Mjallby', 'Mjällby AIF': 'Mjallby',
            'Hammarby': 'Hammarby', 'Varnamo': 'Varnamo',
            'Goteborg': 'Goteborg', 'IFK Göteborg': 'Goteborg',
            'Norrkoping': 'Norrkoping', 'IFK Norrköping': 'Norrkoping',
            'Hacken': 'Hacken', 'BK Häcken': 'Hacken',
            'Brommapojkarna': 'Brommapojkarna', 'IF Brommapojkarna': 'Brommapojkarna',
            'Sirius': 'Sirius', 'IK Sirius': 'Sirius',
            'Elfsborg': 'Elfsborg', 'IF Elfsborg': 'Elfsborg',
            'Halmstad': 'Halmstad', 'Halmstads BK': 'Halmstad',
            'Djurgarden': 'Djurgarden', 'Djurgårdens IF': 'Djurgarden',
            'AIK': 'AIK', 'Degerfors': 'Degerfors'
        }
        
        # Apply team name mapping
        df['HomeTeam'] = df['HomeTeam'].map(team_mapping).fillna(df['HomeTeam'])
        df['AwayTeam'] = df['AwayTeam'].map(team_mapping).fillna(df['AwayTeam'])
        
        # Parse dates if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            # Create a default date if missing
            df['Date'] = pd.Timestamp.now()
        
        # Filter out rows with missing team names
        df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
        df = df[df['HomeTeam'].str.strip() != '']
        df = df[df['AwayTeam'].str.strip() != '']
        
        # Ensure we have the required columns for simulation
        required_columns = ['Date', 'HomeTeam', 'AwayTeam']
        df = df[required_columns]
        
        logger.info(f"Successfully loaded {len(df)} fixtures from {filepath}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading fixtures from {filepath}: {e}")
        return pd.DataFrame()
```

### Fix 3: Update the Fixture Predictions Merge in fix.md

In the `fixture_results_page()` function, update the merge operation:

```python
# Merge with fixture dates - use correct column names
fixture_summary = fixture_summary.merge(
    fixtures_df[['Home_Team', 'Away_Team', 'Date']],  # Changed from HomeTeam/AwayTeam
    left_on=['home_team', 'away_team'],
    right_on=['Home_Team', 'Away_Team'],  # Changed from HomeTeam/AwayTeam
    how='left'
)
```

## Complete Debug Steps

1. **Check file exists**: Verify `data/cleaned/upcoming_fixtures.csv` exists
2. **Fix file path**: Update the path in app.py to match actual location
3. **Update column handling**: Use the enhanced `_load_upcoming_fixtures_directly` method
4. **Test simulation**: Run a small simulation to verify fixtures load correctly
5. **Check predictions merge**: Ensure fixture predictions display correctly

## Test Commands

After making these changes, test with:

```python
# Test fixture loading
from src.simulation.simulator import MonteCarloSimulator
fixtures = MonteCarloSimulator._load_upcoming_fixtures_directly("data/cleaned/upcoming_fixtures.csv")
print(f"Loaded {len(fixtures)} fixtures")
print("Sample fixtures:")
print(fixtures.head())
```

This should resolve the issue where the Monte Carlo simulation isn't getting the upcoming games correctly from your CSV file.