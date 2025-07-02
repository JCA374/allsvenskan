import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def ensure_directory_exists(directory_path):
    """Ensure a directory exists, create if it doesn't"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False

def validate_dataframe(df, required_columns, df_name="DataFrame"):
    """Validate DataFrame structure and content"""
    validation_errors = []
    
    try:
        # Check if DataFrame is empty
        if df.empty:
            validation_errors.append(f"{df_name} is empty")
            return validation_errors
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_errors.append(f"{df_name} missing columns: {missing_columns}")
        
        # Check for null values in required columns
        for col in required_columns:
            if col in df.columns and df[col].isna().any():
                null_count = df[col].isna().sum()
                validation_errors.append(f"{df_name}[{col}] has {null_count} null values")
        
        return validation_errors
        
    except Exception as e:
        return [f"Error validating {df_name}: {str(e)}"]

def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def normalize_team_name(team_name):
    """Normalize team name for consistency"""
    if pd.isna(team_name):
        return team_name
    
    # Basic normalization
    name = str(team_name).strip()
    
    # Common replacements for Swedish teams
    replacements = {
        'IFK Göteborg': 'IFK Goteborg',
        'Malmö FF': 'Malmo FF',
        'Djurgårdens IF': 'Djurgarden',
        'BK Häcken': 'Hacken',
        'Mjällby AIF': 'Mjallby',
        'Östersunds FK': 'Ostersund'
    }
    
    return replacements.get(name, name)

def calculate_points_from_result(home_goals, away_goals):
    """Calculate points awarded for a match result"""
    try:
        home_goals = int(home_goals)
        away_goals = int(away_goals)
        
        if home_goals > away_goals:
            return 3, 0  # Home win
        elif home_goals == away_goals:
            return 1, 1  # Draw
        else:
            return 0, 3  # Away win
            
    except (ValueError, TypeError):
        return 0, 0  # Invalid input

def format_percentage(value, decimal_places=1):
    """Format a decimal as a percentage string"""
    try:
        return f"{value * 100:.{decimal_places}f}%"
    except (TypeError, ValueError):
        return "0.0%"

def get_season_progress(completed_matches, total_matches):
    """Calculate season progress as a percentage"""
    try:
        if total_matches <= 0:
            return 0
        return min(100, (completed_matches / total_matches) * 100)
    except (TypeError, ZeroDivisionError):
        return 0

def parse_match_score(score_string):
    """Parse a score string like '2-1' into home and away goals"""
    try:
        if pd.isna(score_string) or not score_string:
            return None, None
        
        parts = str(score_string).split('-')
        if len(parts) == 2:
            home_goals = int(parts[0].strip())
            away_goals = int(parts[1].strip())
            return home_goals, away_goals
        
        return None, None
        
    except (ValueError, AttributeError):
        return None, None

def create_league_table(results_df):
    """Create current league table from results"""
    try:
        if results_df.empty:
            return pd.DataFrame()
        
        # Get all teams
        home_teams = set(results_df['HomeTeam'].unique())
        away_teams = set(results_df['AwayTeam'].unique())
        all_teams = list(home_teams | away_teams)
        
        # Initialize table
        table_data = []
        
        for team in all_teams:
            # Home matches
            home_matches = results_df[results_df['HomeTeam'] == team]
            away_matches = results_df[results_df['AwayTeam'] == team]
            
            # Calculate stats
            games_played = len(home_matches) + len(away_matches)
            goals_for = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
            goals_against = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
            goal_difference = goals_for - goals_against
            
            # Calculate points
            points = 0
            wins = 0
            draws = 0
            losses = 0
            
            # Home points
            for _, match in home_matches.iterrows():
                home_pts, away_pts = calculate_points_from_result(match['FTHG'], match['FTAG'])
                points += home_pts
                if home_pts == 3:
                    wins += 1
                elif home_pts == 1:
                    draws += 1
                else:
                    losses += 1
            
            # Away points
            for _, match in away_matches.iterrows():
                home_pts, away_pts = calculate_points_from_result(match['FTHG'], match['FTAG'])
                points += away_pts
                if away_pts == 3:
                    wins += 1
                elif away_pts == 1:
                    draws += 1
                else:
                    losses += 1
            
            table_data.append({
                'Team': team,
                'Games': games_played,
                'Wins': wins,
                'Draws': draws,
                'Losses': losses,
                'Goals_For': goals_for,
                'Goals_Against': goals_against,
                'Goal_Difference': goal_difference,
                'Points': points
            })
        
        # Create DataFrame and sort by points, then goal difference
        table_df = pd.DataFrame(table_data)
        table_df = table_df.sort_values(['Points', 'Goal_Difference', 'Goals_For'], 
                                       ascending=[False, False, False])
        table_df['Position'] = range(1, len(table_df) + 1)
        
        return table_df
        
    except Exception as e:
        print(f"Error creating league table: {e}")
        return pd.DataFrame()

def save_data_safely(data, filepath, format='csv'):
    """Safely save data to file with error handling"""
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            ensure_directory_exists(directory)
        
        if format.lower() == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                pd.DataFrame(data).to_csv(filepath, index=False)
        elif format.lower() == 'json':
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        return False

def load_data_safely(filepath, format='csv'):
    """Safely load data from file with error handling"""
    try:
        if not os.path.exists(filepath):
            return None
        
        if format.lower() == 'csv':
            return pd.read_csv(filepath)
        elif format.lower() == 'json':
            import json
            with open(filepath, 'r') as f:
                return json.load(f)
        
        return None
        
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None
