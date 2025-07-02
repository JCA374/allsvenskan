import os
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json

Base = declarative_base()

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.metadata = MetaData()
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            # Matches table for storing raw match data
            self.matches_table = Table(
                'matches', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('date', String),
                Column('venue', String),
                Column('home_team', String),
                Column('away_team', String),
                Column('home_goals', Integer, nullable=True),
                Column('away_goals', Integer, nullable=True),
                Column('match_type', String),  # 'result' or 'fixture'
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Team statistics table
            self.team_stats_table = Table(
                'team_statistics', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('team_name', String),
                Column('total_games', Integer),
                Column('home_games', Integer),
                Column('away_games', Integer),
                Column('total_goals_scored', Integer),
                Column('total_goals_conceded', Integer),
                Column('avg_goals_scored', Float),
                Column('avg_goals_conceded', Float),
                Column('attack_strength', Float),
                Column('defense_strength', Float),
                Column('recent_form', Float),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Model parameters table
            self.model_params_table = Table(
                'model_parameters', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('model_type', String),  # 'poisson'
                Column('home_advantage', Float),
                Column('league_avg', Float),
                Column('attack_rates', String),  # JSON string
                Column('defense_rates', String),  # JSON string
                Column('fitted', Boolean),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Simulation results table
            self.simulation_results_table = Table(
                'simulation_results', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('simulation_id', String),
                Column('n_simulations', Integer),
                Column('team_name', String),
                Column('final_points', Integer),
                Column('simulation_number', Integer),
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            # Analysis results table
            self.analysis_results_table = Table(
                'analysis_results', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('simulation_id', String),
                Column('team_name', String),
                Column('championship_probability', Float),
                Column('relegation_probability', Float),
                Column('european_probability', Float),
                Column('expected_points', Float),
                Column('position_probabilities', String),  # JSON string
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            # Create all tables
            self.metadata.create_all(self.engine)
            print("Database tables created successfully")
            
        except Exception as e:
            print(f"Error creating database tables: {e}")
    
    def save_matches(self, matches_df, match_type='mixed'):
        """Save match data to database"""
        try:
            # Prepare data for insertion
            matches_data = []
            
            for _, row in matches_df.iterrows():
                # Determine if this is a result or fixture based on goals
                if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
                    row_type = 'result'
                    home_goals = int(row['FTHG'])
                    away_goals = int(row['FTAG'])
                else:
                    row_type = 'fixture'
                    home_goals = None
                    away_goals = None
                
                match_data = {
                    'date': str(row.get('Date', '')),
                    'venue': str(row.get('Venue', 'Unknown')),
                    'home_team': str(row.get('HomeTeam', '')),
                    'away_team': str(row.get('AwayTeam', '')),
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'match_type': row_type
                }
                matches_data.append(match_data)
            
            # Insert data
            with self.engine.connect() as conn:
                # Clear existing data
                conn.execute(text("DELETE FROM matches"))
                
                # Insert new data
                conn.execute(self.matches_table.insert(), matches_data)
                conn.commit()
            
            print(f"Saved {len(matches_data)} matches to database")
            return True
            
        except Exception as e:
            print(f"Error saving matches: {e}")
            return False
    
    def load_matches(self, match_type=None):
        """Load match data from database"""
        try:
            query = "SELECT * FROM matches"
            if match_type:
                query += f" WHERE match_type = '{match_type}'"
            
            df = pd.read_sql(query, self.engine)
            
            if not df.empty:
                # Rename columns to match expected format
                df = df.rename(columns={
                    'home_goals': 'FTHG',
                    'away_goals': 'FTAG',
                    'home_team': 'HomeTeam',
                    'away_team': 'AwayTeam',
                    'date': 'Date',
                    'venue': 'Venue'
                })
            
            return df
            
        except Exception as e:
            print(f"Error loading matches: {e}")
            return pd.DataFrame()
    
    def save_team_statistics(self, team_stats_df):
        """Save team statistics to database"""
        try:
            stats_data = []
            
            for team, stats in team_stats_df.iterrows():
                stat_data = {
                    'team_name': team,
                    'total_games': int(stats.get('total_games', 0)),
                    'home_games': int(stats.get('home_games', 0)),
                    'away_games': int(stats.get('away_games', 0)),
                    'total_goals_scored': int(stats.get('total_goals_scored', 0)),
                    'total_goals_conceded': int(stats.get('total_goals_conceded', 0)),
                    'avg_goals_scored': float(stats.get('avg_goals_scored', 0)),
                    'avg_goals_conceded': float(stats.get('avg_goals_conceded', 0)),
                    'attack_strength': float(stats.get('attack_strength', 1.0)),
                    'defense_strength': float(stats.get('defense_strength', 1.0)),
                    'recent_form': float(stats.get('recent_form', 0))
                }
                stats_data.append(stat_data)
            
            with self.engine.connect() as conn:
                # Clear existing data
                conn.execute(text("DELETE FROM team_statistics"))
                
                # Insert new data
                conn.execute(self.team_stats_table.insert(), stats_data)
                conn.commit()
            
            print(f"Saved statistics for {len(stats_data)} teams")
            return True
            
        except Exception as e:
            print(f"Error saving team statistics: {e}")
            return False
    
    def load_team_statistics(self):
        """Load team statistics from database"""
        try:
            df = pd.read_sql("SELECT * FROM team_statistics", self.engine)
            
            if not df.empty:
                df = df.set_index('team_name')
                # Drop database-specific columns
                df = df.drop(columns=['id', 'created_at', 'updated_at'], errors='ignore')
            
            return df
            
        except Exception as e:
            print(f"Error loading team statistics: {e}")
            return pd.DataFrame()
    
    def save_model_parameters(self, model):
        """Save Poisson model parameters to database"""
        try:
            model_data = {
                'model_type': 'poisson',
                'home_advantage': float(model.home_advantage),
                'league_avg': float(model.league_avg),
                'attack_rates': json.dumps(model.attack_rates),
                'defense_rates': json.dumps(model.defense_rates),
                'fitted': bool(model.fitted)
            }
            
            with self.engine.connect() as conn:
                # Clear existing model parameters
                conn.execute(text("DELETE FROM model_parameters WHERE model_type = 'poisson'"))
                
                # Insert new parameters
                conn.execute(self.model_params_table.insert(), [model_data])
                conn.commit()
            
            print("Model parameters saved to database")
            return True
            
        except Exception as e:
            print(f"Error saving model parameters: {e}")
            return False
    
    def load_model_parameters(self):
        """Load Poisson model parameters from database"""
        try:
            query = "SELECT * FROM model_parameters WHERE model_type = 'poisson' ORDER BY created_at DESC LIMIT 1"
            result = pd.read_sql(query, self.engine)
            
            if result.empty:
                return None
            
            row = result.iloc[0]
            
            model_params = {
                'home_advantage': float(row['home_advantage']),
                'league_avg': float(row['league_avg']),
                'attack_rates': json.loads(row['attack_rates']),
                'defense_rates': json.loads(row['defense_rates']),
                'fitted': bool(row['fitted'])
            }
            
            return model_params
            
        except Exception as e:
            print(f"Error loading model parameters: {e}")
            return None
    
    def save_simulation_results(self, simulation_results_df, simulation_id=None):
        """Save simulation results to database"""
        try:
            if simulation_id is None:
                simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare data for insertion
            sim_data = []
            
            for sim_num, (_, sim_result) in enumerate(simulation_results_df.iterrows()):
                for team, points in sim_result.items():
                    sim_data.append({
                        'simulation_id': simulation_id,
                        'n_simulations': len(simulation_results_df),
                        'team_name': team,
                        'final_points': int(points),
                        'simulation_number': sim_num
                    })
            
            with self.engine.connect() as conn:
                # Clear existing simulation results for this ID
                conn.execute(text(f"DELETE FROM simulation_results WHERE simulation_id = '{simulation_id}'"))
                
                # Insert new results
                conn.execute(self.simulation_results_table.insert(), sim_data)
                conn.commit()
            
            print(f"Saved simulation results with ID: {simulation_id}")
            return simulation_id
            
        except Exception as e:
            print(f"Error saving simulation results: {e}")
            return None
    
    def load_simulation_results(self, simulation_id=None):
        """Load simulation results from database"""
        try:
            if simulation_id:
                query = f"SELECT * FROM simulation_results WHERE simulation_id = '{simulation_id}'"
            else:
                # Get the most recent simulation
                query = """
                SELECT * FROM simulation_results 
                WHERE simulation_id = (
                    SELECT simulation_id FROM simulation_results 
                    ORDER BY created_at DESC LIMIT 1
                )
                """
            
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                return pd.DataFrame()
            
            # Pivot the data to get teams as columns
            pivot_df = df.pivot_table(
                index='simulation_number',
                columns='team_name',
                values='final_points',
                aggfunc='first'
            )
            
            return pivot_df
            
        except Exception as e:
            print(f"Error loading simulation results: {e}")
            return pd.DataFrame()
    
    def get_simulation_history(self):
        """Get list of all simulation IDs and their metadata"""
        try:
            query = """
            SELECT simulation_id, n_simulations, MIN(created_at) as created_at, COUNT(*) as total_records
            FROM simulation_results 
            GROUP BY simulation_id, n_simulations
            ORDER BY MIN(created_at) DESC
            """
            
            df = pd.read_sql(query, self.engine)
            return df
            
        except Exception as e:
            print(f"Error getting simulation history: {e}")
            return pd.DataFrame()
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return False