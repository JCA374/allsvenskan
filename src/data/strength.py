import pandas as pd
import numpy as np

class TeamStrengthCalculator:
    def __init__(self):
        self.min_games = 3  # Minimum games to calculate reliable stats
    
    def calculate_strengths(self, results_df):
        """Calculate team strengths from match results"""
        try:
            if results_df.empty:
                return pd.DataFrame()
            
            # Calculate home statistics
            home_stats = results_df.groupby('HomeTeam').agg({
                'FTHG': ['sum', 'mean', 'count'],
                'FTAG': ['sum', 'mean']
            }).round(3)
            
            home_stats.columns = ['home_goals_scored', 'home_goals_avg', 'home_games', 
                                 'home_goals_conceded', 'home_goals_conceded_avg']
            
            # Calculate away statistics
            away_stats = results_df.groupby('AwayTeam').agg({
                'FTAG': ['sum', 'mean', 'count'],
                'FTHG': ['sum', 'mean']
            }).round(3)
            
            away_stats.columns = ['away_goals_scored', 'away_goals_avg', 'away_games',
                                 'away_goals_conceded', 'away_goals_conceded_avg']
            
            # Combine home and away stats
            all_teams = set(results_df['HomeTeam'].unique()) | set(results_df['AwayTeam'].unique())
            team_stats = pd.DataFrame(index=list(all_teams))
            
            # Merge home and away stats
            team_stats = team_stats.join(home_stats, how='left')
            team_stats = team_stats.join(away_stats, how='left')
            
            # Fill NaN values with 0
            team_stats = team_stats.fillna(0)
            
            # Calculate overall statistics
            team_stats['total_games'] = team_stats['home_games'] + team_stats['away_games']
            team_stats['total_goals_scored'] = team_stats['home_goals_scored'] + team_stats['away_goals_scored']
            team_stats['total_goals_conceded'] = team_stats['home_goals_conceded'] + team_stats['away_goals_conceded']
            
            # Calculate averages (avoid division by zero)
            team_stats['avg_goals_scored'] = np.where(
                team_stats['total_games'] > 0,
                team_stats['total_goals_scored'] / team_stats['total_games'],
                0
            )
            
            team_stats['avg_goals_conceded'] = np.where(
                team_stats['total_games'] > 0,
                team_stats['total_goals_conceded'] / team_stats['total_games'],
                0
            )
            
            team_stats['goal_difference'] = team_stats['total_goals_scored'] - team_stats['total_goals_conceded']
            
            # Calculate league averages
            league_avg_goals = results_df[['FTHG', 'FTAG']].mean().mean()
            team_stats['league_avg'] = league_avg_goals
            
            # Calculate attack and defense strengths relative to league average
            team_stats['attack_strength'] = np.where(
                league_avg_goals > 0,
                team_stats['avg_goals_scored'] / league_avg_goals,
                1.0
            )
            
            team_stats['defense_strength'] = np.where(
                league_avg_goals > 0,
                team_stats['avg_goals_conceded'] / league_avg_goals,
                1.0
            )
            
            # Calculate home and away specific strengths
            team_stats['home_attack_strength'] = np.where(
                team_stats['home_games'] >= self.min_games,
                team_stats['home_goals_avg'] / league_avg_goals if league_avg_goals > 0 else 1.0,
                team_stats['attack_strength']
            )
            
            team_stats['away_attack_strength'] = np.where(
                team_stats['away_games'] >= self.min_games,
                team_stats['away_goals_avg'] / league_avg_goals if league_avg_goals > 0 else 1.0,
                team_stats['attack_strength']
            )
            
            team_stats['home_defense_strength'] = np.where(
                team_stats['home_games'] >= self.min_games,
                team_stats['home_goals_conceded_avg'] / league_avg_goals if league_avg_goals > 0 else 1.0,
                team_stats['defense_strength']
            )
            
            team_stats['away_defense_strength'] = np.where(
                team_stats['away_games'] >= self.min_games,
                team_stats['away_goals_conceded_avg'] / league_avg_goals if league_avg_goals > 0 else 1.0,
                team_stats['defense_strength']
            )
            
            # Calculate form (last 5 games)
            team_stats['recent_form'] = team_stats.apply(
                lambda x: self._calculate_recent_form(x.name, results_df), axis=1
            )
            
            # Round all numeric columns
            numeric_columns = team_stats.select_dtypes(include=[np.number]).columns
            team_stats[numeric_columns] = team_stats[numeric_columns].round(3)
            
            return team_stats
            
        except Exception as e:
            print(f"Error calculating team strengths: {e}")
            return pd.DataFrame()
    
    def _calculate_recent_form(self, team, results_df, last_n=5):
        """Calculate recent form for a team (last N games)"""
        try:
            # Get team's recent matches (both home and away)
            home_matches = results_df[results_df['HomeTeam'] == team].tail(last_n//2)
            away_matches = results_df[results_df['AwayTeam'] == team].tail(last_n//2)
            
            points = 0
            games = 0
            
            # Calculate points from home games
            for _, match in home_matches.iterrows():
                games += 1
                if match['FTHG'] > match['FTAG']:  # Win
                    points += 3
                elif match['FTHG'] == match['FTAG']:  # Draw
                    points += 1
            
            # Calculate points from away games
            for _, match in away_matches.iterrows():
                games += 1
                if match['FTAG'] > match['FTHG']:  # Win
                    points += 3
                elif match['FTAG'] == match['FTHG']:  # Draw
                    points += 1
            
            return points / games if games > 0 else 0
            
        except Exception as e:
            return 0
    
    def get_team_summary(self, team_stats):
        """Get a summary of team statistics"""
        if team_stats.empty:
            return {}
        
        summary = {
            'total_teams': len(team_stats),
            'avg_goals_per_game': team_stats['league_avg'].iloc[0] if 'league_avg' in team_stats.columns else 0,
            'highest_scoring': team_stats['avg_goals_scored'].idxmax() if 'avg_goals_scored' in team_stats.columns else None,
            'best_defense': team_stats['avg_goals_conceded'].idxmin() if 'avg_goals_conceded' in team_stats.columns else None,
            'most_games_played': team_stats['total_games'].max() if 'total_games' in team_stats.columns else 0
        }
        
        return summary
