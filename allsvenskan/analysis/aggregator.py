import pandas as pd
import numpy as np

class ResultsAggregator:
    def __init__(self):
        pass
    
    def analyze_results(self, simulation_results_df):
        """Comprehensive analysis of simulation results"""
        try:
            if simulation_results_df.empty:
                return pd.DataFrame()
            
            teams = simulation_results_df.columns.tolist()
            analysis = []
            
            for team in teams:
                team_points = simulation_results_df[team]
                
                team_analysis = {
                    'Team': team,
                    'Mean_Points': round(team_points.mean(), 2),
                    'Median_Points': round(team_points.median(), 2),
                    'Std_Points': round(team_points.std(), 2),
                    'Min_Points': int(team_points.min()),
                    'Max_Points': int(team_points.max()),
                    'Q25_Points': round(team_points.quantile(0.25), 2),
                    'Q75_Points': round(team_points.quantile(0.75), 2)
                }
                
                analysis.append(team_analysis)
            
            analysis_df = pd.DataFrame(analysis)
            return analysis_df.sort_values('Mean_Points', ascending=False)
            
        except Exception as e:
            print(f"Error analyzing results: {e}")
            return pd.DataFrame()
    
    def calculate_position_probabilities(self, simulation_results_df):
        """Calculate probability of each team finishing in each position"""
        try:
            if simulation_results_df.empty:
                return {}
            
            teams = simulation_results_df.columns.tolist()
            n_teams = len(teams)
            position_probs = {}
            
            for team in teams:
                position_probs[team] = [0] * n_teams
            
            # For each simulation, rank teams and count positions
            for _, sim_result in simulation_results_df.iterrows():
                # Sort teams by points (descending)
                ranked_teams = sim_result.sort_values(ascending=False)
                
                for position, (team, points) in enumerate(ranked_teams.items()):
                    position_probs[team][position] += 1
            
            # Convert counts to probabilities
            n_simulations = len(simulation_results_df)
            for team in teams:
                position_probs[team] = [count / n_simulations for count in position_probs[team]]
            
            return position_probs
            
        except Exception as e:
            print(f"Error calculating position probabilities: {e}")
            return {}
    
    def calculate_expected_points(self, simulation_results_df):
        """Calculate expected final points for each team"""
        try:
            if simulation_results_df.empty:
                return {}
            
            expected_points = {}
            for team in simulation_results_df.columns:
                expected_points[team] = round(simulation_results_df[team].mean(), 2)
            
            return expected_points
            
        except Exception as e:
            print(f"Error calculating expected points: {e}")
            return {}
    
    def calculate_championship_odds(self, simulation_results_df):
        """Calculate probability of winning the championship (1st place)"""
        try:
            if simulation_results_df.empty:
                return {}
            
            championship_counts = {}
            teams = simulation_results_df.columns.tolist()
            
            # Initialize counts
            for team in teams:
                championship_counts[team] = 0
            
            # Count championships for each simulation
            for _, sim_result in simulation_results_df.iterrows():
                champion = sim_result.idxmax()  # Team with most points
                championship_counts[champion] += 1
            
            # Convert to probabilities
            n_simulations = len(simulation_results_df)
            championship_odds = {}
            for team in teams:
                championship_odds[team] = championship_counts[team] / n_simulations
            
            return championship_odds
            
        except Exception as e:
            print(f"Error calculating championship odds: {e}")
            return {}
    
    def calculate_relegation_odds(self, simulation_results_df, relegation_spots=3):
        """Calculate probability of relegation (bottom N positions)"""
        try:
            if simulation_results_df.empty:
                return {}
            
            relegation_counts = {}
            teams = simulation_results_df.columns.tolist()
            
            # Initialize counts
            for team in teams:
                relegation_counts[team] = 0
            
            # Count relegations for each simulation
            for _, sim_result in simulation_results_df.iterrows():
                # Get bottom N teams
                relegated_teams = sim_result.nsmallest(relegation_spots).index.tolist()
                
                for team in relegated_teams:
                    relegation_counts[team] += 1
            
            # Convert to probabilities
            n_simulations = len(simulation_results_df)
            relegation_odds = {}
            for team in teams:
                relegation_odds[team] = relegation_counts[team] / n_simulations
            
            return relegation_odds
            
        except Exception as e:
            print(f"Error calculating relegation odds: {e}")
            return {}
    
    def calculate_european_qualification_odds(self, simulation_results_df, european_spots=5):
        """Calculate probability of European qualification (top N positions)"""
        try:
            if simulation_results_df.empty:
                return {}
            
            european_counts = {}
            teams = simulation_results_df.columns.tolist()
            
            # Initialize counts
            for team in teams:
                european_counts[team] = 0
            
            # Count European qualifications for each simulation
            for _, sim_result in simulation_results_df.iterrows():
                # Get top N teams
                european_teams = sim_result.nlargest(european_spots).index.tolist()
                
                for team in european_teams:
                    european_counts[team] += 1
            
            # Convert to probabilities
            n_simulations = len(simulation_results_df)
            european_odds = {}
            for team in teams:
                european_odds[team] = european_counts[team] / n_simulations
            
            return european_odds
            
        except Exception as e:
            print(f"Error calculating European qualification odds: {e}")
            return {}
    
    def generate_final_table_prediction(self, simulation_results_df):
        """Generate most likely final table based on expected points"""
        try:
            expected_points = self.calculate_expected_points(simulation_results_df)
            
            # Sort teams by expected points
            final_table = sorted(expected_points.items(), key=lambda x: x[1], reverse=True)
            
            table_df = pd.DataFrame(final_table, columns=['Team', 'Expected_Points'])
            table_df['Position'] = range(1, len(table_df) + 1)
            
            # Add additional statistics
            position_probs = self.calculate_position_probabilities(simulation_results_df)
            championship_odds = self.calculate_championship_odds(simulation_results_df)
            relegation_odds = self.calculate_relegation_odds(simulation_results_df)
            
            table_df['Championship_Prob'] = table_df['Team'].map(
                lambda x: round(championship_odds.get(x, 0) * 100, 2)
            )
            table_df['Relegation_Prob'] = table_df['Team'].map(
                lambda x: round(relegation_odds.get(x, 0) * 100, 2)
            )
            
            return table_df
            
        except Exception as e:
            print(f"Error generating final table prediction: {e}")
            return pd.DataFrame()
    
    def save_analysis_report(self, simulation_results_df, output_dir="reports"):
        """Save comprehensive analysis report"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate all analyses
            analysis = self.analyze_results(simulation_results_df)
            final_table = self.generate_final_table_prediction(simulation_results_df)
            position_probs = self.calculate_position_probabilities(simulation_results_df)
            
            # Save to CSV files
            analysis.to_csv(f"{output_dir}/team_analysis.csv", index=False)
            final_table.to_csv(f"{output_dir}/final_table_prediction.csv", index=False)
            
            # Save position probabilities
            position_df = pd.DataFrame(position_probs).T
            position_df.columns = [f"Pos_{i+1}" for i in range(len(position_df.columns))]
            position_df.to_csv(f"{output_dir}/position_probabilities.csv")
            
            print(f"Analysis reports saved to {output_dir}/")
            
        except Exception as e:
            print(f"Error saving analysis report: {e}")
