# Fix Guide: Column Name Standardization for Swedish Football Prediction System

## Executive Summary
The football prediction system is experiencing a critical data merge failure causing fixtures from July 5th, 2025 to disappear from predictions. The root cause is **inconsistent column naming conventions** across the data pipeline, particularly between `Home_Team/Away_Team` (with underscores) and `HomeTeam/AwayTeam` (without underscores).

## The Core Problem

### Missing Fixtures
Three games scheduled for 2025-07-05 exist in the data but fail to display:
- GAIS vs Malmo FF (15:00)
- Oster vs Mjallby (15:00)
- Hammarby vs Varnamo (17:30)

### Root Cause Analysis
The merge operation in `app.py` fails due to column name mismatch:

```python
# Current FAILING code in app.py:
fixture_summary = fixture_summary.merge(
    fixtures_df[['Home_Team', 'Away_Team', 'Date']],  # Source columns
    left_on=['home_team', 'away_team'],
    right_on=['HomeTeam', 'AwayTeam'],  # ❌ WRONG! Looking for non-existent columns
    how='left'
)
```

## System-Wide Analysis

### Column Name Inventory

1. **upcoming_fixtures.csv**: Uses `Home_Team`, `Away_Team`
2. **results.csv**: Uses `HomeTeam`, `AwayTeam` 
3. **fixture_predictions.csv**: Uses `home_team`, `away_team` (lowercase)
4. **scraper.py**: Creates `HomeTeam`, `AwayTeam`
5. **simulator.py**: Expects `HomeTeam`, `AwayTeam` but handles conversion
6. **fixtures_cleaner.py**: Outputs `HomeTeam`, `AwayTeam`

### The Inconsistency Chain
```
Data Source → Column Format → Used By
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Football-Data API → HomeTeam/AwayTeam → scraper.py
fixtures_cleaner.py → HomeTeam/AwayTeam → simulator.py
upcoming_fixtures.csv → Home_Team/Away_Team → app.py (FAILS HERE)
Monte Carlo output → home_team/away_team → fixture_predictions.csv
```

## Comprehensive Solution

### Step 1: Immediate Fix (Quick Patch)

Update the merge operation in `app.py` (line ~287):

```python
# Fix the merge operation to use correct column names
fixture_summary = fixture_summary.merge(
    fixtures_df[['Home_Team', 'Away_Team', 'Date']],
    left_on=['home_team', 'away_team'],
    right_on=['Home_Team', 'Away_Team'],  # ✅ FIXED: Match actual column names
    how='left'
)
```

### Step 2: Standardization Strategy (Robust Solution)

Implement a single column naming convention across the entire system:

**Chosen Standard**: `HomeTeam` and `AwayTeam` (no underscores, PascalCase)

#### Why This Standard?
- Already used by Football-Data API (primary data source)
- Consistent with Python naming conventions for data columns
- Used by most internal processing modules

### Step 3: Implementation Plan

#### 3.1 Create Column Name Standardizer

Create `src/utils/column_standardizer.py`:

```python
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
```

#### 3.2 Update Data Pipeline Components

**Update scraper.py** (already correct):
```python
# No changes needed - already outputs HomeTeam/AwayTeam
```

**Update fixtures_cleaner.py** to ensure standardized output:
```python
# In clean_fixtures_file method, before returning:
from src.utils.column_standardizer import ColumnStandardizer

# ... existing code ...
fixtures_df = pd.DataFrame(fixtures)
fixtures_df = ColumnStandardizer.standardize_columns(fixtures_df)
return fixtures_df
```

**Update simulator.py** to handle any format:
```python
# At the beginning of load_fixtures method:
from src.utils.column_standardizer import ColumnStandardizer

def load_fixtures(self, fixtures_path: str) -> pd.DataFrame:
    """Load and standardize fixtures"""
    df = pd.read_csv(fixtures_path)
    df = ColumnStandardizer.standardize_columns(df)

    # Validate required columns
    required = ['HomeTeam', 'AwayTeam', 'Date']
    if not ColumnStandardizer.validate_required_columns(df, required):
        raise ValueError(f"Fixtures file missing required columns: {required}")

    return df
```

**Update app.py** fixture loading:
```python
# In fixture_results_page function:
from src.utils.column_standardizer import ColumnStandardizer

# Load upcoming fixtures
fixtures_df = pd.read_csv("data/clean/upcoming_fixtures.csv", parse_dates=['Date'])
fixtures_df = ColumnStandardizer.standardize_columns(fixtures_df)

# Update merge operation to use standardized names
fixture_summary = fixture_summary.merge(
    fixtures_df[['HomeTeam', 'AwayTeam', 'Date']],
    left_on=['home_team', 'away_team'],
    right_on=['HomeTeam', 'AwayTeam'],
    how='left'
)
```

### Step 4: Data Migration Script

Create `scripts/migrate_column_names.py`:

```python
import os
import pandas as pd
from src.utils.column_standardizer import ColumnStandardizer

def migrate_csv_files():
    """Migrate all CSV files to use standardized column names"""

    csv_files = [
        'data/clean/upcoming_fixtures.csv',
        'data/clean/results.csv',
        'data/processed/historical_results.csv',
        'reports/simulations/fixture_predictions.csv'
    ]

    for filepath in csv_files:
        if os.path.exists(filepath):
            print(f"Migrating {filepath}...")

            # Backup original
            backup_path = filepath.replace('.csv', '_backup.csv')
            df = pd.read_csv(filepath)
            df.to_csv(backup_path, index=False)

            # Standardize and save
            df_standard = ColumnStandardizer.standardize_columns(df)
            df_standard.to_csv(filepath, index=False)

            print(f"✓ Migrated {filepath}")
            print(f"  Backup saved to {backup_path}")

if __name__ == "__main__":
    migrate_csv_files()
```

### Step 5: Testing & Validation

Create `tests/test_column_standardization.py`:

```python
import pandas as pd
import pytest
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

    # Merge should work now
    merged = predictions.merge(
        fixtures_std,
        left_on=['home_team', 'away_team'],
        right_on=['HomeTeam', 'AwayTeam'],
        how='left'
    )

    assert len(merged) == 2
    assert 'Date' in merged.columns
```

## Maintenance Guidelines

### 1. **Always Use the Standardizer**
   - Import `ColumnStandardizer` when loading any CSV
   - Apply standardization before any merge operations

### 2. **Document Column Names**
   - Add column descriptions to data dictionary
   - Update README with expected formats

### 3. **Monitor for New Variations**
   - Log warnings when unknown column names appear
   - Update `STANDARD_COLUMNS` mapping as needed

### 4. **Regular Validation**
   ```python
   # Add to daily maintenance script
   python scripts/validate_data_consistency.py
   ```

## Prevention Strategy

### Pre-commit Hook
Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Check for non-standard column names in CSV files
grep -r "Home_Team\|Away_Team\|home_team\|away_team" data/clean/*.csv && {
    echo "⚠️  WARNING: Non-standard column names detected!"
    echo "Run: python scripts/migrate_column_names.py"
}
```

### CI/CD Pipeline Check
```yaml
# .github/workflows/data-validation.yml
- name: Validate Column Names
  run: |
    python -m pytest tests/test_column_standardization.py
    python scripts/validate_column_consistency.py
```

## Rollback Plan

If issues arise after implementation:

1. **Restore from backups**:
   ```bash
   for f in data/*_backup.csv; do
     mv "$f" "${f/_backup/}"
   done
   ```

2. **Revert code changes**:
   ```bash
   git revert HEAD~5..HEAD
   ```

3. **Apply quick fix only** (Step 1)

## Success Metrics

After implementation, verify:
- ✅ All July 5th fixtures appear in predictions
- ✅ No merge warnings in logs
- ✅ All tests pass
- ✅ Simulation runs without column errors
- ✅ Historical data integrity maintained

## Timeline

1. **Immediate** (5 min): Apply Step 1 quick fix
2. **Today** (2 hours): Implement standardizer and update core files
3. **Tomorrow** (1 hour): Run migration and full testing
4. **This Week**: Monitor logs and fix edge cases
5. **Next Sprint**: Add automated validation to CI/CD

---

**Remember**: The goal is not just to fix the immediate issue but to prevent it from happening again through robust, maintainable design.