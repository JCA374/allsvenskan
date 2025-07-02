# Monte Carlo Simulation Fixture Error Fix

## Error Description

**Error Message:** `No upcoming fixtures found in data sources. Creating realistic remaining season fixtures.`

## Root Cause Analysis

This error occurs when the Monte Carlo simulation tries to run but cannot find any upcoming fixtures in the `data/clean/fixtures.csv` file. This typically happens due to one of the following reasons:

1. **Empty fixtures file**: The data scraping/cleaning process didn't capture any upcoming matches
2. **All fixtures are completed**: The season is complete and no remaining matches exist
3. **Data processing pipeline failure**: The fixtures weren't properly separated from results during cleaning
4. **File path issues**: The fixtures file doesn't exist or is in the wrong location

## Current Behavior

When this error occurs, the system automatically creates "realistic remaining season fixtures" as a fallback mechanism. However, this creates artificial fixtures rather than using real match data, which reduces prediction accuracy.

## Step-by-Step Fix

### 1. Verify Data Sources

First, check if your data files exist and contain data:

```bash
# Check if files exist
ls -la data/clean/
ls -la data/raw/

# Check file contents
head -10 data/clean/fixtures.csv
head -10 data/clean/results.csv
wc -l data/clean/fixtures.csv
```

### 2. Re-run Data Collection

If fixtures are missing, re-scrape the data:

```python
# In your Streamlit app, go to "Data Collection" page
# Click "🕷️ Scrape Latest Data" button
# Ensure you select current season (2024/2025)
```

### 3. Fix Data Cleaning Process

The issue might be in the data cleaning where results and fixtures aren't properly separated. Update your cleaning logic:

```python
# In src/data/clean.py or equivalent
def separate_results_and_fixtures(df):
    """Properly separate completed matches from upcoming fixtures"""
    current_date = pd.Timestamp.now()
    
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Separate based on whether match has results
    completed_matches = df[
        (df['HomeGoals'].notna()) & 
        (df['AwayGoals'].notna()) &
        (df['Date'] <= current_date)
    ].copy()
    
    upcoming_fixtures = df[
        (df['HomeGoals'].isna()) | 
        (df['AwayGoals'].isna()) |
        (df['Date'] > current_date)
    ].copy()
    
    # Clean upcoming fixtures
    upcoming_fixtures = upcoming_fixtures.drop(['HomeGoals', 'AwayGoals'], axis=1, errors='ignore')
    
    return completed_matches, upcoming_fixtures
```

### 4. Manual Fixture Creation (Temporary Fix)

If scraping fails, create fixtures manually based on Allsvenskan schedule:

```python
def create_manual_fixtures():
    """Create manual fixtures for remaining season"""
    import itertools
    from datetime import datetime, timedelta
    
    # Get teams from results
    results = pd.read_csv("data/clean/results.csv")
    teams = sorted(set(results['HomeTeam'].unique()).union(set(results['AwayTeam'].unique())))
    
    # Allsvenskan teams (2024 season)
    allsvenskan_teams = [
        'AIK', 'BK Häcken', 'Djurgårdens IF', 'Elfsborg', 'GAIS', 
        'Göteborg', 'Hammarby', 'Halmstads BK', 'Kalmar FF', 
        'Malmö FF', 'Mjällby AIF', 'Sirius', 'Värnamo', 
        'Västerås SK', 'BP', 'Brommapojkarna'
    ]
    
    # Use scraped teams or fallback to known teams
    if len(teams) == 0:
        teams = allsvenskan_teams
    
    fixtures = []
    start_date = datetime.now() + timedelta(days=7)
    
    # Create remaining round-robin fixtures
    for round_num in range(1, 4):  # Assuming 3 rounds remaining
        round_fixtures = list(itertools.combinations(teams, 2))
        
        for i, (home, away) in enumerate(round_fixtures[:len(teams)//2]):
            fixture_date = start_date + timedelta(days=round_num*14 + i)
            
            fixtures.append({
                'Date': fixture_date,
                'HomeTeam': home,
                'AwayTeam': away,
                'Round': f'Round {round_num}'
            })
    
    return pd.DataFrame(fixtures)
```

### 5. Update Simulation Logic

Modify the simulation page to handle empty fixtures more gracefully:

```python
def simulation_page():
    st.header("🎲 Monte Carlo Simulation")
    
    # Check prerequisites
    if not st.session_state.model_trained and not os.path.exists("models/poisson_params.pkl"):
        st.warning("⚠️ Please train the model first in the Model Training section.")
        return
    
    # Load fixtures with error handling
    try:
        fixtures = pd.read_csv("data/clean/fixtures.csv")
        
        if len(fixtures) == 0:
            st.warning("⚠️ No upcoming fixtures found. Options:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Re-scrape Data"):
                    # Trigger data re-scraping
                    st.rerun()
            
            with col2:
                if st.button("🔧 Create Sample Fixtures"):
                    fixtures = create_manual_fixtures()
                    fixtures.to_csv("data/clean/fixtures.csv", index=False)
                    st.success(f"Created {len(fixtures)} sample fixtures")
                    st.rerun()
            
            return
            
    except FileNotFoundError:
        st.error("❌ Fixtures file not found. Please run data collection first.")
        return
    
    # Continue with simulation...
```

### 6. Data Validation

Add validation to ensure data quality:

```python
def validate_fixtures(fixtures_df):
    """Validate fixtures data before simulation"""
    issues = []
    
    if fixtures_df.empty:
        issues.append("Fixtures dataframe is empty")
    
    required_columns = ['Date', 'HomeTeam', 'AwayTeam']
    missing_cols = [col for col in required_columns if col not in fixtures_df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    if 'Date' in fixtures_df.columns:
        try:
            pd.to_datetime(fixtures_df['Date'])
        except:
            issues.append("Date column contains invalid dates")
    
    null_teams = fixtures_df[['HomeTeam', 'AwayTeam']].isnull().sum().sum()
    if null_teams > 0:
        issues.append(f"Found {null_teams} null team values")
    
    return issues
```

### 7. Configuration Update

Update your app configuration to handle this scenario:

```python
# In app.py or main configuration
SIMULATION_CONFIG = {
    'min_fixtures_required': 5,
    'create_sample_fixtures': True,
    'sample_fixture_rounds': 3,
    'validate_before_simulation': True
}
```

## Prevention Strategy

### 1. Regular Data Updates
- Schedule automatic data scraping
- Monitor data file sizes and content
- Set up alerts for empty data files

### 2. Robust Error Handling
- Implement comprehensive try-catch blocks
- Provide meaningful error messages
- Offer recovery options to users

### 3. Data Validation Pipeline
- Validate data at each processing step
- Check for minimum required fixtures
- Ensure data consistency across files

### 4. Fallback Mechanisms
- Maintain backup fixture data
- Implement sample fixture generation
- Provide manual data entry options

## Testing the Fix

1. **Delete existing fixtures**: `rm data/clean/fixtures.csv`
2. **Run data collection**: Use the scraping functionality
3. **Verify fixtures exist**: Check file has content
4. **Run simulation**: Should work without the error
5. **Test fallback**: Delete fixtures again and ensure graceful handling

