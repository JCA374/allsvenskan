# Fix: Import Current Standings and Add Simulation Scores

## 1. Update `src/utils/helpers.py` - Add function to calculate current standings

```python
def calculate_current_standings(results_df):
    """Calculate current league standings from completed matches"""
    try:
        if results_df.empty:
            return {}
        
        # Get all teams
        home_teams = set(results_df['HomeTeam'].unique())
        away_teams = set(results_df['AwayTeam'].unique())
        all_teams = list(home_teams | away_teams)
        
        # Initialize standings
        standings = {}
        for team in all_teams:
            standings[team] = {
                'played': 0,
                'won': 0,
                'drawn': 0,
                'lost': 0,
                'goals_for': 0,
                'goals_against': 0,
                'goal_diff': 0,
                'points': 0
            }
        
        # Process each match
        for _, match in results_df.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            home_goals = int(match['FTHG']) if pd.notna(match['FTHG']) else 0
            away_goals = int(match['FTAG']) if pd.notna(match['FTAG']) else 0
            
            # Update games played
            standings[home_team]['played'] += 1
            standings[away_team]['played'] += 1
            
            # Update goals
            standings[home_team]['goals_for'] += home_goals
            standings[home_team]['goals_against'] += away_goals
            standings[away_team]['goals_for'] += away_goals
            standings[away_team]['goals_against'] += home_goals
            
            # Update results and points
            if home_goals > away_goals:
                standings[home_team]['won'] += 1
                standings[home_team]['points'] += 3
                standings[away_team]['lost'] += 1
            elif home_goals < away_goals:
                standings[away_team]['won'] += 1
                standings[away_team]['points'] += 3
                standings[home_team]['lost'] += 1
            else:
                standings[home_team]['drawn'] += 1
                standings[away_team]['drawn'] += 1
                standings[home_team]['points'] += 1
                standings[away_team]['points'] += 1
        
        # Calculate goal difference
        for team in standings:
            standings[team]['goal_diff'] = standings[team]['goals_for'] - standings[team]['goals_against']
        
        return standings
        
    except Exception as e:
        print(f"Error calculating standings: {e}")
        return {}

def get_current_points_table(standings):
    """Extract just team:points from full standings"""
    return {team: data['points'] for team, data in standings.items()}
```

## 2. Update `src/simulation/simulator.py` - Add methods for simulating with current standings

```python
def simulate_season_with_current_standings(self, current_standings):
    """Simulate a season starting from current standings"""
    try:
        # Start with current points
        points_table = current_standings.copy()
        
        # Ensure all teams are in the table
        for team in self.teams:
            if team not in points_table:
                points_table[team] = 0
        
        # Simulate only remaining fixtures
        for _, fixture in self.fixtures.iterrows():
            home_team = fixture['HomeTeam']
            away_team = fixture['AwayTeam']
            
            # Skip if teams don't exist in our model
            if home_team not in self.teams or away_team not in self.teams:
                continue
            
            mu_home, mu_away = self.model.predict_match(home_team, away_team)
            
            home_goals = self.rng.poisson(mu_home)
            away_goals = self.rng.poisson(mu_away)
            
            if home_goals > away_goals:
                points_table[home_team] += 3
            elif home_goals == away_goals:
                points_table[home_team] += 1
                points_table[away_team] += 1
            else:
                points_table[away_team] += 3
        
        return points_table
    
    except Exception as e:
        print(f"Error simulating with current standings: {e}")
        return current_standings

def simulate_remaining_fixtures_detailed(self, n_simulations=1000):
    """Simulate remaining fixtures and return detailed results"""
    try:
        fixture_results = []
        
        for sim_idx in range(n_simulations):
            sim_fixtures = []
            
            for _, fixture in self.fixtures.iterrows():
                home_team = fixture['HomeTeam']
                away_team = fixture['AwayTeam']
                
                if home_team not in self.teams or away_team not in self.teams:
                    continue
                
                mu_home, mu_away = self.model.predict_match(home_team, away_team)
                
                home_goals = self.rng.poisson(mu_home)
                away_goals = self.rng.poisson(mu_away)
                
                sim_fixtures.append({
                    'simulation': sim_idx,
                    'date': fixture.get('Date', ''),
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'home_win': 1 if home_goals > away_goals else 0,
                    'draw': 1 if home_goals == away_goals else 0,
                    'away_win': 1 if home_goals < away_goals else 0
                })
            
            fixture_results.extend(sim_fixtures)
        
        return pd.DataFrame(fixture_results)
    
    except Exception as e:
        print(f"Error simulating fixtures: {e}")
        return pd.DataFrame()

def run_monte_carlo_with_standings(self, n_simulations, current_standings, progress_callback=None):
    """Run Monte Carlo simulations starting from current standings"""
    try:
        print(f"Starting {n_simulations:,} simulations with current standings...")
        simulation_results = []
        
        batch_size = min(1000, n_simulations // 10)
        
        for i in range(n_simulations):
            if i % batch_size == 0 and progress_callback:
                progress = (i / n_simulations) * 100
                progress_callback(progress)
            
            season_result = self.simulate_season_with_current_standings(current_standings)
            simulation_results.append(season_result)
        
        results_df = pd.DataFrame(simulation_results)
        
        # Ensure all teams are present
        for team in self.teams:
            if team not in results_df.columns:
                results_df[team] = current_standings.get(team, 0)
        
        print(f"Completed {n_simulations:,} simulations with current standings!")
        return results_df
        
    except Exception as e:
        print(f"Error running simulations with standings: {e}")
        return pd.DataFrame()
```

## 3. Update `src/data/scraper.py` - Separate completed matches from fixtures

```python
def separate_results_and_fixtures(self, df):
    """Separate completed matches from upcoming fixtures"""
    try:
        # Completed matches have scores
        results = df[
            df['FTHG'].notna() & 
            df['FTAG'].notna() & 
            (df['FTHG'] != '') & 
            (df['FTAG'] != '')
        ].copy()
        
        # Fixtures don't have scores yet
        fixtures = df[
            df['FTHG'].isna() | 
            df['FTAG'].isna() | 
            (df['FTHG'] == '') | 
            (df['FTAG'] == '')
        ].copy()
        
        print(f"Separated {len(results)} completed matches and {len(fixtures)} fixtures")
        
        return results, fixtures
        
    except Exception as e:
        print(f"Error separating results and fixtures: {e}")
        return df[df['FTHG'].notna()], df[df['FTHG'].isna()]
```

## 4. Update `app.py` - Add current standings integration

```python
# In the simulation page function, add:

def simulation_page():
    st.header("🎲 Season Simulation")
    
    # Load data
    try:
        results_df = pd.read_csv("data/clean/results.csv", parse_dates=['Date'])
        fixtures_df = pd.read_csv("data/clean/fixtures.csv", parse_dates=['Date'])
        
        # Calculate current standings
        from src.utils.helpers import calculate_current_standings, get_current_points_table
        
        current_standings_full = calculate_current_standings(results_df)
        current_points = get_current_points_table(current_standings_full)
        
        # Display current standings
        st.subheader("📊 Current League Table")
        
        if current_standings_full:
            standings_df = pd.DataFrame.from_dict(current_standings_full, orient='index')
            standings_df = standings_df.sort_values(['points', 'goal_diff', 'goals_for'], 
                                                  ascending=False)
            standings_df.reset_index(inplace=True)
            standings_df.rename(columns={'index': 'Team'}, inplace=True)
            standings_df.index = range(1, len(standings_df) + 1)
            
            st.dataframe(
                standings_df[['Team', 'played', 'won', 'drawn', 'lost', 
                             'goals_for', 'goals_against', 'goal_diff', 'points']],
                use_container_width=True
            )
        
        st.info(f"📅 {len(results_df)} matches completed, {len(fixtures_df)} fixtures remaining")
        
        # Simulation settings
        st.subheader("⚙️ Simulation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_current_standings = st.checkbox(
                "Start from current standings", 
                value=True,
                help="If checked, simulations will start from current league positions"
            )
            
            n_simulations = st.slider(
                "Number of Simulations",
                min_value=100,
                max_value=50000,
                value=10000,
                step=100
            )
        
        with col2:
            if st.button("🚀 Run Simulation", type="primary"):
                try:
                    with st.spinner(f"Running {n_simulations:,} simulations..."):
                        progress_bar = st.progress(0)
                        
                        # Initialize simulator
                        simulator = MonteCarloSimulator(
                            results=results_df,
                            fixtures=fixtures_df,
                            model=load_model()
                        )
                        
                        # Run simulations
                        if use_current_standings:
                            simulation_results = simulator.run_monte_carlo_with_standings(
                                n_simulations=n_simulations,
                                current_standings=current_points,
                                progress_callback=lambda p: progress_bar.progress(int(p))
                            )
                        else:
                            simulation_results = simulator.run_monte_carlo(
                                n_simulations=n_simulations,
                                progress_callback=lambda p: progress_bar.progress(int(p))
                            )
                        
                        # Save results
                        simulation_results.to_csv("reports/simulations/sim_results.csv", index=False)
                        
                        # Also save fixture predictions
                        fixture_predictions = simulator.simulate_remaining_fixtures_detailed(
                            n_simulations=min(1000, n_simulations)
                        )
                        fixture_predictions.to_csv("reports/simulations/fixture_predictions.csv", 
                                                 index=False)
                        
                        st.success("✅ Simulation completed!")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

# Add new fixture results page
def fixture_results_page():
    st.header("⚽ Fixture Predictions")
    
    if not os.path.exists("reports/simulations/fixture_predictions.csv"):
        st.warning("⚠️ Please run simulations first.")
        return
    
    try:
        # Load fixture predictions
        predictions_df = pd.read_csv("reports/simulations/fixture_predictions.csv")
        fixtures_df = pd.read_csv("data/clean/fixtures.csv", parse_dates=['Date'])
        
        # Group by match
        fixture_summary = predictions_df.groupby(['home_team', 'away_team']).agg({
            'home_win': 'mean',
            'draw': 'mean',
            'away_win': 'mean',
            'home_goals': 'mean',
            'away_goals': 'mean'
        }).reset_index()
        
        # Merge with fixture dates
        fixture_summary = fixture_summary.merge(
            fixtures_df[['HomeTeam', 'AwayTeam', 'Date']],
            left_on=['home_team', 'away_team'],
            right_on=['HomeTeam', 'AwayTeam'],
            how='left'
        )
        
        # Sort by date
        fixture_summary = fixture_summary.sort_values('Date')
        
        # Display fixtures by date
        st.subheader("📅 Upcoming Fixtures with Predictions")
        
        # Date filter
        unique_dates = fixture_summary['Date'].dt.date.unique()
        selected_date = st.selectbox("Select Date", options=['All'] + list(unique_dates))
        
        if selected_date != 'All':
            display_df = fixture_summary[fixture_summary['Date'].dt.date == selected_date]
        else:
            display_df = fixture_summary
        
        # Format for display
        for _, match in display_df.iterrows():
            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col1:
                st.write(f"**{match['Date'].strftime('%Y-%m-%d')}**")
            
            with col2:
                st.write(f"{match['home_team']} vs {match['away_team']}")
                
                # Probability bars
                probs = {
                    'Home Win': match['home_win'] * 100,
                    'Draw': match['draw'] * 100,
                    'Away Win': match['away_win'] * 100
                }
                
                # Create mini bar chart
                fig = go.Figure(data=[
                    go.Bar(x=list(probs.keys()), y=list(probs.values()),
                          text=[f"{v:.1f}%" for v in probs.values()],
                          textposition='auto')
                ])
                fig.update_layout(height=200, showlegend=False, 
                                margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                st.metric("Expected Score", 
                         f"{match['home_goals']:.1f} - {match['away_goals']:.1f}")
                
                # Most likely result
                if match['home_win'] > max(match['draw'], match['away_win']):
                    st.success("Home Win")
                elif match['away_win'] > max(match['draw'], match['home_win']):
                    st.info("Away Win")
                else:
                    st.warning("Draw")
            
            st.divider()
        
    except Exception as e:
        st.error(f"❌ Error displaying fixture predictions: {str(e)}")
```

## 5. Update main app navigation to include fixture results

```python
# In main() function, add:

pages = {
    "🏠 Overview": overview_page,
    "📥 Data Collection": data_collection_page,
    "✅ Data Verification": data_verification_page,
    "🎲 Simulation": simulation_page,
    "📈 Results Analysis": analysis_page,
    "⚽ Fixture Predictions": fixture_results_page,  # Add this
    "📊 Interactive Dashboard": dashboard_page,
    "🗄️ Database Management": database_page
}
```

## 6. Create visualization for remaining fixtures

```python
# Add to src/visualization/dashboard.py

def create_fixture_timeline(fixture_predictions, fixtures_df):
    """Create timeline visualization of remaining fixtures"""
    try:
        # Merge predictions with fixture info
        timeline_data = fixture_predictions.groupby(['date', 'home_team', 'away_team']).agg({
            'home_win': 'mean',
            'draw': 'mean',
            'away_win': 'mean'
        }).reset_index()
        
        # Create Gantt-style chart
        fig = go.Figure()
        
        for idx, match in timeline_data.iterrows():
            # Determine color based on most likely outcome
            if match['home_win'] > max(match['draw'], match['away_win']):
                color = 'green'
                result = 'Home Win'
            elif match['away_win'] > max(match['draw'], match['home_win']):
                color = 'red'
                result = 'Away Win'
            else:
                color = 'yellow'
                result = 'Draw'
            
            fig.add_trace(go.Scatter(
                x=[match['date'], match['date']],
                y=[idx, idx],
                mode='markers+text',
                marker=dict(size=15, color=color),
                text=f"{match['home_team']} vs {match['away_team']}<br>{result}",
                textposition="top center",
                showlegend=False
            ))
        
        fig.update_layout(
            title="Remaining Fixtures Timeline",
            xaxis_title="Date",
            yaxis_title="Match",
            height=600
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating fixture timeline: {e}")
        return go.Figure()
```

## Key Features Added:

1. **Current Standings Calculation**: Properly calculates current league table from completed matches
2. **Simulation from Current Position**: Simulations can start from actual current standings
3. **Fixture Predictions**: Detailed predictions for each remaining match
4. **Visualization**: Shows probabilities and expected scores for each fixture
5. **Timeline View**: Visual representation of remaining fixtures

## Usage:

1. Run data collection to get latest results and fixtures
2. Go to Simulation page - it will show current standings
3. Run simulation with "Start from current standings" checked
4. View fixture predictions in the new "Fixture Predictions" page
5. Analyze overall season predictions in Results Analysis

The system now properly:
- Calculates current standings from completed matches
- Starts simulations from these standings
- Shows detailed predictions for each remaining fixture
- Provides probability distributions for match outcomes