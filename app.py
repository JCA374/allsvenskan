import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text

# Import custom modules
from src.data.scraper import AllsvenskanScraper
from src.data.cleaner import DataCleaner
from src.data.strength import TeamStrengthCalculator
from src.models.poisson_model import PoissonModel
from src.simulation.simulator import MonteCarloSimulator
from src.analysis.aggregator import ResultsAggregator
from src.visualization.dashboard import Dashboard
from src.database.db_manager import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="Allsvenskan Monte Carlo Forecast",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'simulation_complete' not in st.session_state:
    st.session_state.simulation_complete = False
if 'db_manager' not in st.session_state:
    try:
        st.session_state.db_manager = DatabaseManager()
        st.session_state.db_connected = st.session_state.db_manager.test_connection()
    except Exception as e:
        st.session_state.db_manager = None
        st.session_state.db_connected = False

def main():
    st.title("⚽ Allsvenskan Monte Carlo Forecast")
    st.markdown("### Predicting Swedish Football League Outcomes Using Statistical Modeling")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Collection", "Data Verification", "Model Training", "Monte Carlo Simulation", "Results Analysis", "Dashboard", "Database Management"]
    )
    
    if page == "Data Collection":
        data_collection_page()
    elif page == "Data Verification":
        data_verification_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Monte Carlo Simulation":
        simulation_page()
    elif page == "Results Analysis":
        analysis_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Database Management":
        database_management_page()

def data_collection_page():
    st.header("📊 Data Collection & Processing")
    
    # Database status indicator
    if st.session_state.get('db_connected', False):
        st.success("🗄️ Database connected and ready")
    else:
        st.warning("⚠️ Database not connected - using file storage only")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Web Scraping")
        
        # Year selection
        st.write("**Select Years to Scrape:**")
        
        # Quick selection buttons
        col_quick1, col_quick2 = st.columns(2)
        with col_quick1:
            if st.button("Last 3 Years", type="secondary"):
                st.session_state.selected_years = [2023, 2024, 2025]
        with col_quick2:
            if st.button("Last 10 Years", type="secondary"):
                st.session_state.selected_years = list(range(2015, 2026))
        
        # Initialize selected years if not already set
        if 'selected_years' not in st.session_state:
            st.session_state.selected_years = [2025]
        
        # Multi-select for years
        available_years = list(range(2015, 2026))
        selected_years = st.multiselect(
            "Custom year selection:",
            options=available_years,
            default=st.session_state.selected_years,
            key="year_selector"
        )
        
        if selected_years:
            st.session_state.selected_years = selected_years
        
        # Display selection info
        if st.session_state.selected_years:
            st.info(f"📅 Will scrape: {', '.join(map(str, sorted(st.session_state.selected_years)))}")
        
        if st.button("🔍 Scrape Data from Selected Years", type="primary", disabled=not st.session_state.selected_years):
            with st.spinner(f"Scraping data from {len(st.session_state.selected_years)} years..."):
                try:
                    scraper = AllsvenskanScraper()
                    raw_data = scraper.scrape_matches(years=st.session_state.selected_years)
                    
                    if raw_data is not None and not raw_data.empty:
                        # Save raw data to files
                        os.makedirs("data/raw", exist_ok=True)
                        raw_data.to_csv("data/raw/fixtures_results_raw.csv", index=False)
                        years_str = ', '.join(map(str, sorted(st.session_state.selected_years)))
                        st.success(f"✅ Successfully scraped {len(raw_data)} matches from {len(st.session_state.selected_years)} years ({years_str})!")
                        st.dataframe(raw_data.head())
                        
                        # Clean data automatically
                        cleaner = DataCleaner()
                        results, fixtures = cleaner.clean_data(raw_data)
                        
                        # Save to files
                        os.makedirs("data/clean", exist_ok=True)
                        results.to_csv("data/clean/results.csv", index=False)
                        fixtures.to_csv("data/clean/fixtures.csv", index=False)
                        
                        # Save to database if connected
                        if st.session_state.get('db_connected', False):
                            try:
                                # Combine results and fixtures for database storage
                                combined_data = pd.concat([results, fixtures], ignore_index=True)
                                if st.session_state.db_manager.save_matches(combined_data):
                                    st.success("✅ Data saved to database")
                                else:
                                    st.warning("⚠️ Could not save to database - using file storage")
                            except Exception as e:
                                st.warning(f"⚠️ Database save failed: {str(e)}")
                        
                        st.session_state.data_loaded = True
                        st.success(f"✅ Data cleaned: {len(results)} completed matches, {len(fixtures)} upcoming fixtures")
                        
                    else:
                        st.error("❌ Failed to scrape data. Please check the website or try again later.")
                        
                except Exception as e:
                    st.error(f"❌ Error during scraping: {str(e)}")
    
    with col2:
        st.subheader("Data Status")
        
        # Check database first, then files
        db_data_available = False
        if st.session_state.get('db_connected', False):
            try:
                db_results = st.session_state.db_manager.load_matches('result')
                db_fixtures = st.session_state.db_manager.load_matches('fixture')
                
                if not db_results.empty or not db_fixtures.empty:
                    st.success("✅ Data available in database")
                    st.metric("Completed Matches (DB)", len(db_results))
                    st.metric("Upcoming Fixtures (DB)", len(db_fixtures))
                    db_data_available = True
                    st.session_state.data_loaded = True
                    
                    # Show recent results from database
                    if len(db_results) > 0:
                        st.subheader("Recent Results (Database)")
                        recent = db_results.tail(5)
                        for _, match in recent.iterrows():
                            st.write(f"{match['HomeTeam']} {match['FTHG']}-{match['FTAG']} {match['AwayTeam']}")
            except Exception as e:
                st.warning(f"⚠️ Database error: {str(e)}")
        
        # Fallback to file data if database not available
        if not db_data_available:
            results_exist = os.path.exists("data/clean/results.csv")
            fixtures_exist = os.path.exists("data/clean/fixtures.csv")
            
            if results_exist and fixtures_exist:
                st.success("✅ Clean data files found")
                results = pd.read_csv("data/clean/results.csv")
                fixtures = pd.read_csv("data/clean/fixtures.csv")
                
                st.metric("Completed Matches (Files)", len(results))
                st.metric("Upcoming Fixtures (Files)", len(fixtures))
                
                st.session_state.data_loaded = True
                
                # Show recent results
                if len(results) > 0:
                    st.subheader("Recent Results (Files)")
                    recent = results.tail(5)
                    for _, match in recent.iterrows():
                        st.write(f"{match['HomeTeam']} {match['FTHG']}-{match['FTAG']} {match['AwayTeam']}")
            else:
                st.warning("⚠️ No data found. Please scrape data first.")

def data_verification_page():
    st.header("🔍 Data Verification")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Collection section.")
        return
    
    st.markdown("Verify that your data has been loaded and cleaned correctly by examining specific matches.")
    
    # Load data from database or files
    try:
        # Try database first
        all_matches = None
        if st.session_state.get('db_connected', False):
            try:
                all_matches = st.session_state.db_manager.load_matches()
                data_source = "Database"
            except Exception as e:
                st.warning(f"Database load failed: {str(e)}")
        
        # Fallback to files
        if all_matches is None or all_matches.empty:
            results_file = "data/clean/results.csv"
            fixtures_file = "data/clean/fixtures.csv"
            
            if os.path.exists(results_file) and os.path.exists(fixtures_file):
                results = pd.read_csv(results_file)
                fixtures = pd.read_csv(fixtures_file)
                all_matches = pd.concat([results, fixtures], ignore_index=True)
                data_source = "Files"
            else:
                st.error("❌ No data found. Please collect data first.")
                return
        
        if all_matches.empty:
            st.error("❌ No match data available.")
            return
        
        st.success(f"✅ Data loaded from {data_source}")
        
        # Parse dates to extract years
        try:
            all_matches['Date'] = pd.to_datetime(all_matches['Date'], errors='coerce')
            all_matches['Year'] = all_matches['Date'].dt.year
        except:
            # If date parsing fails, try to extract year from string
            all_matches['Year'] = all_matches['Date'].astype(str).str.extract(r'(\d{4})').astype(float)
        
        # Get available teams and years
        all_teams = sorted(set(all_matches['HomeTeam'].dropna().unique()) | 
                          set(all_matches['AwayTeam'].dropna().unique()))
        available_years = sorted(all_matches['Year'].dropna().unique(), reverse=True)
        
        # Create verification interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_home_team = st.selectbox(
                "Select Home Team:",
                options=all_teams,
                index=0 if all_teams else None
            )
        
        with col2:
            selected_away_team = st.selectbox(
                "Select Away Team:",
                options=all_teams,
                index=1 if len(all_teams) > 1 else 0
            )
        
        with col3:
            selected_year = st.selectbox(
                "Select Year:",
                options=available_years,
                index=0 if available_years else None
            )
        
        # Filter and display results
        if st.button("🔍 Search Matches", type="primary"):
            # Filter matches based on selection
            filtered_matches = all_matches[
                (
                    ((all_matches['HomeTeam'] == selected_home_team) & 
                     (all_matches['AwayTeam'] == selected_away_team)) |
                    ((all_matches['HomeTeam'] == selected_away_team) & 
                     (all_matches['AwayTeam'] == selected_home_team))
                ) &
                (all_matches['Year'] == selected_year)
            ].copy()
            
            if not filtered_matches.empty:
                st.subheader(f"📋 Matches: {selected_home_team} vs {selected_away_team} in {int(selected_year)}")
                
                # Separate results and fixtures
                results = filtered_matches[filtered_matches['FTHG'].notna() & filtered_matches['FTAG'].notna()]
                fixtures = filtered_matches[filtered_matches['FTHG'].isna() | filtered_matches['FTAG'].isna()]
                
                # Display results
                if not results.empty:
                    st.write(f"**Completed Matches ({len(results)}):**")
                    
                    display_results = results[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
                    display_results['Score'] = display_results['FTHG'].astype(int).astype(str) + " - " + display_results['FTAG'].astype(int).astype(str)
                    display_results['Result'] = display_results.apply(
                        lambda x: f"{x['HomeTeam']} {x['Score']} {x['AwayTeam']}", axis=1
                    )
                    
                    for _, match in display_results.iterrows():
                        st.write(f"• {match['Date'].strftime('%Y-%m-%d') if pd.notna(match['Date']) else 'Unknown date'}: {match['Result']}")
                
                # Display fixtures
                if not fixtures.empty:
                    st.write(f"**Upcoming Fixtures ({len(fixtures)}):**")
                    
                    display_fixtures = fixtures[['Date', 'HomeTeam', 'AwayTeam']].copy()
                    
                    for _, match in display_fixtures.iterrows():
                        st.write(f"• {match['Date'].strftime('%Y-%m-%d') if pd.notna(match['Date']) else 'Unknown date'}: {match['HomeTeam']} vs {match['AwayTeam']}")
                
                # Show data quality summary
                st.subheader("📊 Data Quality Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Matches Found", len(filtered_matches))
                
                with col2:
                    st.metric("Completed Results", len(results))
                
                with col3:
                    st.metric("Upcoming Fixtures", len(fixtures))
                
                with col4:
                    missing_dates = filtered_matches['Date'].isna().sum()
                    st.metric("Missing Dates", missing_dates)
                
                # Show raw data for verification
                if st.checkbox("Show Raw Data"):
                    st.subheader("Raw Match Data")
                    st.dataframe(filtered_matches)
                
            else:
                st.info(f"ℹ️ No matches found between {selected_home_team} and {selected_away_team} in {int(selected_year)}")
                
                # Show alternative suggestions
                st.subheader("💡 Suggestions")
                
                # Show matches involving either team in that year
                team_matches = all_matches[
                    ((all_matches['HomeTeam'] == selected_home_team) | 
                     (all_matches['AwayTeam'] == selected_home_team) |
                     (all_matches['HomeTeam'] == selected_away_team) | 
                     (all_matches['AwayTeam'] == selected_away_team)) &
                    (all_matches['Year'] == selected_year)
                ]
                
                if not team_matches.empty:
                    st.write(f"Found {len(team_matches)} matches involving these teams in {int(selected_year)}:")
                    
                    # Show unique opponents
                    opponents_home = set(team_matches[team_matches['HomeTeam'] == selected_home_team]['AwayTeam'].dropna())
                    opponents_away = set(team_matches[team_matches['AwayTeam'] == selected_home_team]['HomeTeam'].dropna())
                    opponents_home2 = set(team_matches[team_matches['HomeTeam'] == selected_away_team]['AwayTeam'].dropna())
                    opponents_away2 = set(team_matches[team_matches['AwayTeam'] == selected_away_team]['HomeTeam'].dropna())
                    
                    all_opponents = (opponents_home | opponents_away | opponents_home2 | opponents_away2) - {selected_home_team, selected_away_team}
                    
                    if all_opponents:
                        st.write("Teams they played against:")
                        for opponent in sorted(all_opponents):
                            st.write(f"• {opponent}")
        
        # Overall data summary
        st.subheader("📈 Overall Data Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_matches = len(all_matches)
            completed_matches = len(all_matches[all_matches['FTHG'].notna() & all_matches['FTAG'].notna()])
            st.metric("Total Matches in Dataset", total_matches)
            st.metric("Completed Matches", completed_matches)
        
        with col2:
            st.metric("Total Teams", len(all_teams))
            st.metric("Years Covered", len(available_years))
        
        with col3:
            if available_years:
                st.metric("Earliest Year", int(min(available_years)))
                st.metric("Latest Year", int(max(available_years)))
        
        # Show team list
        if st.checkbox("Show All Teams"):
            st.subheader("All Teams in Dataset")
            teams_per_row = 3
            for i in range(0, len(all_teams), teams_per_row):
                cols = st.columns(teams_per_row)
                for j, team in enumerate(all_teams[i:i+teams_per_row]):
                    with cols[j]:
                        st.write(f"• {team}")
    
    except Exception as e:
        st.error(f"❌ Error during data verification: {str(e)}")
        st.info("Please check that your data has been collected and cleaned properly.")

def model_training_page():
    st.header("🧠 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Collection section.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team Strength Analysis")
        
        if st.button("📈 Calculate Team Strengths", type="primary"):
            with st.spinner("Calculating team strengths..."):
                try:
                    # Load results from database first, then files
                    results = None
                    if st.session_state.get('db_connected', False):
                        try:
                            results = st.session_state.db_manager.load_matches('result')
                            st.info("📊 Using data from database")
                        except Exception as e:
                            st.warning(f"⚠️ Database load failed: {str(e)}")
                    
                    if results is None or results.empty:
                        results = pd.read_csv("data/clean/results.csv")
                        st.info("📁 Using data from files")
                    
                    strength_calc = TeamStrengthCalculator()
                    team_stats = strength_calc.calculate_strengths(results)
                    
                    # Save to files
                    os.makedirs("data/processed", exist_ok=True)
                    team_stats.to_csv("data/processed/team_stats.csv")
                    
                    # Save to database if connected
                    if st.session_state.get('db_connected', False):
                        try:
                            if st.session_state.db_manager.save_team_statistics(team_stats):
                                st.success("✅ Team strengths saved to database")
                        except Exception as e:
                            st.warning(f"⚠️ Database save failed: {str(e)}")
                    
                    st.success("✅ Team strengths calculated!")
                    st.dataframe(team_stats)
                    
                except Exception as e:
                    st.error(f"❌ Error calculating strengths: {str(e)}")
    
    with col2:
        st.subheader("Poisson Model Training")
        
        if st.button("⚙️ Train Poisson Model", type="primary"):
            if not os.path.exists("data/processed/team_stats.csv"):
                st.error("❌ Please calculate team strengths first.")
                return
                
            with st.spinner("Training Poisson model..."):
                try:
                    results = pd.read_csv("data/clean/results.csv")
                    team_stats = pd.read_csv("data/processed/team_stats.csv", index_col=0)
                    
                    model = PoissonModel()
                    model.fit(results, team_stats)
                    
                    os.makedirs("models", exist_ok=True)
                    model.save("models/poisson_params.pkl")
                    
                    st.session_state.model_trained = True
                    st.success("✅ Poisson model trained successfully!")
                    
                    # Show model parameters
                    st.subheader("Model Parameters")
                    st.metric("Home Advantage Factor", f"{model.home_advantage:.3f}")
                    st.metric("League Average Goals/Game", f"{model.league_avg:.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")

def simulation_page():
    st.header("🎲 Monte Carlo Simulation")
    
    if not st.session_state.model_trained and not os.path.exists("models/poisson_params.pkl"):
        st.warning("⚠️ Please train the model first in the Model Training section.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simulation Parameters")
        
        n_simulations = st.slider("Number of Simulations", 1000, 50000, 10000, step=1000)
        random_seed = st.number_input("Random Seed", value=42, step=1)
        
        if st.button("🎯 Run Monte Carlo Simulation", type="primary"):
            with st.spinner(f"Running {n_simulations:,} simulations..."):
                try:
                    fixtures = pd.read_csv("data/clean/fixtures.csv")
                    
                    # Check if we have any fixtures to simulate
                    if len(fixtures) == 0:
                        st.info("📅 No upcoming fixtures found in data sources. Creating realistic remaining season fixtures...")
                        
                        # Get current teams and create realistic remaining fixtures
                        results = pd.read_csv("data/clean/results.csv")
                        teams = sorted(set(results['HomeTeam'].unique()).union(set(results['AwayTeam'].unique())))
                        
                        # Create realistic remaining season fixtures
                        from datetime import datetime, timedelta
                        import itertools
                        
                        sample_fixtures = []
                        start_date = datetime.now() + timedelta(days=7)
                        
                        # Create round-robin style fixtures (more realistic)
                        fixture_count = 0
                        for i, (home, away) in enumerate(itertools.combinations(teams, 2)):
                            if fixture_count >= 30:  # Limit to reasonable number
                                break
                            
                            # Add home and away fixtures
                            sample_fixtures.append({
                                'Date': start_date + timedelta(days=fixture_count * 3),  # Every 3 days
                                'HomeTeam': home,
                                'AwayTeam': away
                            })
                            fixture_count += 1
                            
                            if fixture_count < 30:
                                sample_fixtures.append({
                                    'Date': start_date + timedelta(days=fixture_count * 3),
                                    'HomeTeam': away,
                                    'AwayTeam': home
                                })
                                fixture_count += 1
                        
                        fixtures = pd.DataFrame(sample_fixtures)
                        st.success(f"✅ Created {len(fixtures)} realistic remaining season fixtures for {len(teams)} teams")
                    
                    model = PoissonModel()
                    model.load("models/poisson_params.pkl")
                    
                    simulator = MonteCarloSimulator(fixtures, model, seed=random_seed)
                    simulation_results = simulator.run(n_simulations)
                    
                    os.makedirs("reports/simulations", exist_ok=True)
                    simulation_results.to_csv("reports/simulations/sim_results.csv", index=False)
                    
                    st.session_state.simulation_complete = True
                    st.success(f"✅ Completed {n_simulations:,} simulations!")
                    
                    # Show progress summary
                    aggregator = ResultsAggregator()
                    summary = aggregator.analyze_results(simulation_results)
                    
                    st.subheader("Simulation Summary")
                    st.dataframe(summary.head(10))
                    
                except Exception as e:
                    st.error(f"❌ Error running simulation: {str(e)}")
    
    with col2:
        st.subheader("Simulation Status")
        
        if os.path.exists("reports/simulations/sim_results.csv"):
            st.success("✅ Simulation results available")
            
            sim_results = pd.read_csv("reports/simulations/sim_results.csv")
            st.metric("Simulations Completed", len(sim_results))
            st.metric("Teams Analyzed", len(sim_results.columns))
            
            # Show sample simulation outcomes
            st.subheader("Sample Final Tables (First 3 Simulations)")
            for i in range(min(3, len(sim_results))):
                st.write(f"**Simulation {i+1}:**")
                sim_table = sim_results.iloc[i].sort_values(ascending=False)
                for j, (team, points) in enumerate(sim_table.head(5).items()):
                    st.write(f"{j+1}. {team}: {points} pts")
                st.write("---")
        else:
            st.info("ℹ️ No simulation results found. Run simulation above.")

def analysis_page():
    st.header("📈 Results Analysis")
    
    if not os.path.exists("reports/simulations/sim_results.csv"):
        st.warning("⚠️ Please run simulations first.")
        return
    
    try:
        sim_results = pd.read_csv("reports/simulations/sim_results.csv")
        aggregator = ResultsAggregator()
        
        # Generate comprehensive analysis
        position_probs = aggregator.calculate_position_probabilities(sim_results)
        expected_points = aggregator.calculate_expected_points(sim_results)
        championship_odds = aggregator.calculate_championship_odds(sim_results)
        relegation_odds = aggregator.calculate_relegation_odds(sim_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Championship Probabilities")
            champ_df = pd.DataFrame({
                'Team': [team for team, prob in championship_odds.items()],
                'Championship %': [prob * 100 for team, prob in championship_odds.items()]
            })
            champ_df = champ_df.sort_values('Championship %', ascending=False)
            
            fig_champ = px.bar(champ_df.head(8), x='Team', y='Championship %',
                              title="Top 8 Championship Contenders")
            fig_champ.update_xaxes(tickangle=45)
            st.plotly_chart(fig_champ, use_container_width=True)
            
        with col2:
            st.subheader("⬇️ Relegation Probabilities")
            releg_df = pd.DataFrame({
                'Team': [team for team, prob in relegation_odds.items()],
                'Relegation %': [prob * 100 for team, prob in relegation_odds.items()]
            })
            releg_df = releg_df.sort_values('Relegation %', ascending=False)
            
            fig_releg = px.bar(releg_df.head(8), x='Team', y='Relegation %',
                              title="Top 8 Relegation Candidates", color_discrete_sequence=['red'])
            fig_releg.update_xaxes(tickangle=45)
            st.plotly_chart(fig_releg, use_container_width=True)
        
        # Position probability heatmap
        st.subheader("📊 Position Probability Matrix")
        
        # Convert position probabilities to DataFrame for heatmap
        pos_matrix = pd.DataFrame(position_probs).T
        pos_matrix.columns = [f"Pos {i}" for i in range(1, len(pos_matrix.columns) + 1)]
        
        fig_heatmap = px.imshow(pos_matrix.values, 
                               x=pos_matrix.columns,
                               y=pos_matrix.index,
                               color_continuous_scale='RdYlBu_r',
                               title="Final Position Probabilities")
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Expected points comparison
        st.subheader("📏 Expected Final Points")
        points_df = pd.DataFrame({
            'Team': [team for team, points in expected_points.items()],
            'Expected Points': [points for team, points in expected_points.items()]
        })
        points_df = points_df.sort_values('Expected Points', ascending=True)
        
        fig_points = px.bar(points_df, x='Expected Points', y='Team', 
                           orientation='h', title="Expected Final League Points")
        st.plotly_chart(fig_points, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error analyzing results: {str(e)}")

def dashboard_page():
    st.header("📋 Interactive Dashboard")
    
    if not os.path.exists("reports/simulations/sim_results.csv"):
        st.warning("⚠️ Please run simulations first to view dashboard.")
        return
    
    try:
        dashboard = Dashboard()
        dashboard.create_full_dashboard()
        
    except Exception as e:
        st.error(f"❌ Error loading dashboard: {str(e)}")

def database_management_page():
    st.header("🗄️ Database Management")
    
    if not st.session_state.get('db_connected', False):
        st.error("❌ Database not connected")
        st.info("Database functionality requires a PostgreSQL connection. Please check your environment variables.")
        return
    
    st.success("✅ Database connected successfully")
    
    # Database status overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Database Overview")
        
        try:
            # Check table counts
            with st.session_state.db_manager.engine.connect() as conn:
                matches_count = conn.execute(text("SELECT COUNT(*) FROM matches")).scalar()
                team_stats_count = conn.execute(text("SELECT COUNT(*) FROM team_statistics")).scalar()
                model_params_count = conn.execute(text("SELECT COUNT(*) FROM model_parameters")).scalar()
                sim_results_count = conn.execute(text("SELECT COUNT(DISTINCT simulation_id) FROM simulation_results")).scalar()
            
            st.metric("Total Matches", matches_count)
            st.metric("Team Statistics Records", team_stats_count)
            st.metric("Model Parameters", model_params_count)
            st.metric("Simulation Runs", sim_results_count)
            
        except Exception as e:
            st.error(f"Error querying database: {str(e)}")
    
    with col2:
        st.subheader("🔧 Database Operations")
        
        if st.button("📋 View Recent Matches"):
            try:
                recent_matches = st.session_state.db_manager.load_matches()
                if not recent_matches.empty:
                    st.dataframe(recent_matches.head(10))
                else:
                    st.info("No matches found in database")
            except Exception as e:
                st.error(f"Error loading matches: {str(e)}")
        
        if st.button("📈 View Team Statistics"):
            try:
                team_stats = st.session_state.db_manager.load_team_statistics()
                if not team_stats.empty:
                    st.dataframe(team_stats)
                else:
                    st.info("No team statistics found in database")
            except Exception as e:
                st.error(f"Error loading team statistics: {str(e)}")
        
        if st.button("🎲 View Simulation History"):
            try:
                sim_history = st.session_state.db_manager.get_simulation_history()
                if not sim_history.empty:
                    st.dataframe(sim_history)
                else:
                    st.info("No simulation history found")
            except Exception as e:
                st.error(f"Error loading simulation history: {str(e)}")
    
    # Advanced operations
    st.subheader("⚠️ Advanced Operations")
    st.warning("Use these operations carefully as they may affect stored data")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🔄 Test Database Connection", type="secondary"):
            if st.session_state.db_manager.test_connection():
                st.success("✅ Database connection successful")
            else:
                st.error("❌ Database connection failed")
    
    with col4:
        if st.button("📊 Show Database Schema", type="secondary"):
            st.subheader("Database Tables")
            tables_info = [
                ("matches", "Stores match data (results and fixtures)"),
                ("team_statistics", "Team performance statistics"),
                ("model_parameters", "Poisson model parameters"),
                ("simulation_results", "Monte Carlo simulation results"),
                ("analysis_results", "Aggregated analysis results")
            ]
            
            for table_name, description in tables_info:
                st.write(f"**{table_name}**: {description}")

if __name__ == "__main__":
    main()
