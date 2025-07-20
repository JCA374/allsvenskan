import os
import pandas as pd
import sys

# Add src directory to path so we can import ColumnStandardizer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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

            try:
                # Backup original
                backup_path = filepath.replace('.csv', '_backup.csv')
                df = pd.read_csv(filepath)
                df.to_csv(backup_path, index=False)

                # Standardize and save
                df_standard = ColumnStandardizer.standardize_columns(df)
                df_standard.to_csv(filepath, index=False)

                print(f"✓ Migrated {filepath}")
                print(f"  Backup saved to {backup_path}")

            except Exception as e:
                print(f"✗ Error migrating {filepath}: {e}")
        else:
            print(f"⚠ File not found: {filepath}")

if __name__ == "__main__":
    migrate_csv_files()