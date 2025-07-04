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
from src.data.odds_api import OddsAPI
from src.data.odds_schema import OddsData, OddsRecord
from src.models.hybrid_model import HybridPoissonOddsModel
from src.utils.odds_converter import remove_margin, calculate_margin

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
        ["Data Collection", "Data Verification", "Model Training", "Odds Integration", "Monte Carlo Simulation", "Fixture Predictions", "Results Analysis", "Dashboard", "Database Management"]
    )

    if page == "Data Collection":
        data_collection_page()
    elif page == "Data Verification":
        data_verification_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Odds Integration":
        odds_integration_page()
    elif page == "Monte Carlo Simulation":
        simulation_page()
    elif page == "Fixture Predictions":
        fixture_results_page()
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

        col2a, col2b = st.columns(2)

        with col2a:
            if st.button("⚙️ Fast Training", type="primary", help="Faster training without MLE optimization"):
                if not os.path.exists("data/processed/team_stats.csv"):
                    st.error("❌ Please calculate team strengths first.")
                    return

                with st.spinner("Training Poisson model (fast mode)..."):
                    try:
                        results = pd.read_csv("data/clean/results.csv")
                        team_stats = pd.read_csv("data/processed/team_stats.csv", index_col=0)

                        model = PoissonModel()
                        model.fit(results, team_stats)

                        os.makedirs("models", exist_ok=True)
                        model.save("models/poisson_params.pkl")

                        # Store model in session state for hybrid use
                        st.session_state.poisson_model = model
                        st.session_state.model_trained = True
                        st.success("✅ Poisson model trained successfully (fast mode)!")

                        # Show model parameters
                        st.subheader("Model Parameters")
                        st.metric("Home Advantage Factor", f"{model.home_advantage:.3f}")
                        st.metric("League Average Goals/Game", f"{model.league_avg:.3f}")

                    except Exception as e:
                        st.error(f"❌ Error training model: {str(e)}")

        with col2b:
            if st.button("🎯 Advanced Training", help="Slower but more accurate with MLE optimization"):
                if not os.path.exists("data/processed/team_stats.csv"):
                    st.error("❌ Please calculate team strengths first.")
                    return

                with st.spinner("Training Poisson model (advanced mode)..."):
                    try:
                        results = pd.read_csv("data/clean/results.csv")
                        team_stats = pd.read_csv("data/processed/team_stats.csv", index_col=0)

                        # Advanced model with MLE and Dixon-Coles
                        model = PoissonModel(use_mle=True, use_dixon_coles=True)
                        model.fit(results, team_stats)

                        os.makedirs("models", exist_ok=True)
                        model.save("models/poisson_params.pkl")

                        # Store model in session state for hybrid use
                        st.session_state.poisson_model = model
                        st.session_state.model_trained = True
                        st.success("✅ Poisson model trained successfully (advanced mode)!")

                        # Show model parameters
                        st.subheader("Model Parameters")
                        st.metric("Home Advantage Factor", f"{model.home_advantage:.3f}")
                        st.metric("League Average Goals/Game", f"{model.league_avg:.3f}")
                    except Exception as e:
                        st.error(f"❌ Error training model: {str(e)}")

def simulation_page():
    st.header("🎲 Season Simulation")

    # Load data
    try:
        results_df = pd.read_csv("data/clean/results.csv", parse_dates=['Date'])

        # Check for upcoming fixtures file and use it if available
        upcoming_fixtures_path = "data/clean/upcoming_fixtures.csv"
        if os.path.exists(upcoming_fixtures_path):
            st.info("📅 Using upcoming_fixtures.csv for simulation")
            fixtures_source = "upcoming_fixtures"
            # Will load cleaned fixtures in simulation section
            fixtures_df = None
        else:
            fixtures_df = pd.read_csv("data/clean/fixtures.csv", parse_dates=['Date'])
            fixtures_source = "generated_fixtures"
            st.info("📅 Using generated fixtures for simulation")

        # Calculate current standings from live data
        from src.utils.helpers import calculate_current_standings_from_url, calculate_current_standings, get_current_points_table

        # Try to get live standings first, fallback to local data
        current_standings_df = calculate_current_standings_from_url()

        if current_standings_df.empty:
            st.warning("Using local data for standings calculation")
            current_standings_full = calculate_current_standings(results_df)
            current_points = get_current_points_table(current_standings_full)
        else:
            st.info("Using live Football-Data for current standings")
            # Convert DataFrame to points dictionary for simulation
            current_points = dict(zip(current_standings_df['Team'], current_standings_df['Pts']))

        # Display current standings
        st.subheader("📊 Current League Table")

        if not current_standings_df.empty:
            # Use live data format
            standings_df = current_standings_df.copy()
            standings_df.index = range(1, len(standings_df) + 1)

            st.dataframe(
                standings_df[['Team', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']],
                use_container_width=True
            )
        elif 'current_standings_full' in locals() and current_standings_full:
            # Use local data format (fallback)
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
        else:
            st.warning("No standings data available")

        # Show fixtures information
        if fixtures_source == "upcoming_fixtures":
            # Load and display info about upcoming fixtures using direct method
            from src.simulation.simulator import MonteCarloSimulator
            temp_fixtures = MonteCarloSimulator._load_upcoming_fixtures_directly(upcoming_fixtures_path)
            if not temp_fixtures.empty:
                st.info(f"📅 {len(results_df)} matches completed, {len(temp_fixtures)} upcoming fixtures loaded from upcoming_fixtures.csv")

                # Show sample upcoming fixtures
                if st.checkbox("Show upcoming fixtures preview"):
                    st.subheader("📅 Next Upcoming Fixtures")
                    next_fixtures = temp_fixtures.head(10)
                    for _, fixture in next_fixtures.iterrows():
                        date_str = fixture['Date'].strftime('%Y-%m-%d') if hasattr(fixture['Date'], 'strftime') else str(fixture['Date'])
                        st.write(f"• {date_str}: {fixture['HomeTeam']} vs {fixture['AwayTeam']}")
            else:
                st.error("Could not load upcoming fixtures from upcoming_fixtures.csv")
                st.info("Please ensure the upcoming_fixtures.csv file is properly formatted")
        else:
            st.warning("No upcoming_fixtures.csv file found - please add authentic fixture data to run simulations")

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
            # Hybrid model settings
            use_odds_integration = st.checkbox(
                "Use Odds Integration", 
                value=st.session_state.get('use_odds_integration', False),
                help="Combine Poisson model with betting odds for more accurate predictions"
            )
            st.session_state.use_odds_integration = use_odds_integration

            if use_odds_integration:
                season_progress = st.slider(
                    "Season Progress",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get('season_progress', 0.5),
                    step=0.05,
                    help="0.0 = Season start (more odds weight), 1.0 = Season end (more Poisson weight)"
                )
                st.session_state.season_progress = season_progress
            else:
                season_progress = 0.5

        with col2:
            if st.button("🚀 Run Simulation", type="primary"):
                try:
                    with st.spinner(f"Running {n_simulations:,} simulations..."):
                        progress_bar = st.progress(0)

                        # Initialize model and simulator
                        from src.models.poisson_model import PoissonModel
                        from src.simulation.simulator import MonteCarloSimulator

                        model = PoissonModel()
                        if os.path.exists("models/poisson_params.pkl"):
                            model.load("models/poisson_params.pkl")
                        else:
                            st.warning("Model not found, training new one...")
                            team_stats = pd.read_csv("data/processed/team_statistics.csv")
                            model.fit(results_df, team_stats)
                            os.makedirs("models", exist_ok=True)
                            model.save("models/poisson_params.pkl")

                        # Initialize simulator based on fixtures source
                        if fixtures_source == "upcoming_fixtures":
                            try:
                                # Create hybrid model if odds integration is enabled
                                hybrid_model = None
                                odds_data = None

                                if use_odds_integration and st.session_state.get('odds_fetched', False):
                                    from src.models.hybrid_model import HybridPoissonOddsModel
                                    hybrid_model = HybridPoissonOddsModel(model)
                                    odds_data = st.session_state.get('odds_data')
                                    st.info("🔮 Using Hybrid Model (Poisson + Odds)")
                                else:
                                    st.info("📊 Using Pure Poisson Model")

                                simulator = MonteCarloSimulator.from_upcoming_fixtures(
                                    model, upcoming_fixtures_path, seed=42
                                )

                                # Add hybrid model support to existing simulator
                                simulator.hybrid_model = hybrid_model
                                simulator.odds_data = odds_data
                                simulator.season_progress = season_progress

                                st.success("✅ Using upcoming_fixtures.csv for simulation")
                            except Exception as e:
                                st.error(f"❌ Could not load upcoming fixtures: {e}")
                                st.stop()  # Stop execution since no fixtures are available
                        else:
                            st.error("❌ No upcoming_fixtures.csv file found")
                            st.info("Please add the upcoming_fixtures.csv file to run simulations")
                            st.stop()

                        # Run simulations
                        if use_current_standings:
                            simulation_results = simulator.run_monte_carlo_with_standings(
                                n_simulations=n_simulations,
                                current_standings=current_points,
                                progress_callback=lambda p: progress_bar.progress(p / 100)
                            )
                        else:
                            simulation_results = simulator.run(
                                n_simulations=n_simulations,
                                progress_callback=lambda p: progress_bar.progress(p / 100)
                            )

                        # Save results
                        os.makedirs("reports/simulations", exist_ok=True)
                        simulation_results.to_csv("reports/simulations/sim_results.csv", index=False)

                        # Also save fixture predictions
                        fixture_predictions = simulator.simulate_remaining_fixtures_detailed(
                            n_simulations=min(1000, n_simulations)
                        )
                        fixture_predictions.to_csv("reports/simulations/fixture_predictions.csv", 
                                                 index=False)

                        st.success("✅ Simulation completed!")
                        st.session_state.simulation_complete = True

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # Show simulation status
        if os.path.exists("reports/simulations/sim_results.csv"):
            st.subheader("📈 Simulation Results")
            sim_results = pd.read_csv("reports/simulations/sim_results.csv")
            st.metric("Simulations Completed", len(sim_results))
            st.metric("Teams Analyzed", len(sim_results.columns))

            # Show expected final table (combining current + simulated points)
            expected_points = sim_results.mean().sort_values(ascending=False)
            st.write("**Expected Final Season Table:**")
            for i, (team, points) in enumerate(expected_points.head(len(expected_points)).items()):
                current_pts = current_points.get(team, 0)
                simulated_pts = points - current_pts if points >= current_pts else points
                st.write(f"{i+1}. {team}: {points:.1f} pts (Current: {current_pts}, +{simulated_pts:.1f} from remaining fixtures)")

            # Show current vs projected standings comparison
            if not current_standings_df.empty and len(current_standings_df) > 0:
                st.subheader("📈 Standings Projection")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Current Position**")
                    for i, row in current_standings_df.head(10).iterrows():
                        st.write(f"{i+1}. {row['Team']}: {row['Pts']} pts")

                with col2:
                    st.write("**Projected Final Position**")
                    for i, (team, points) in enumerate(expected_points.head(10).items()):
                        st.write(f"{i+1}. {team}: {points:.1f} pts")

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

def fixture_results_page():
    st.header("⚽ Fixture Predictions")

    if not os.path.exists("reports/simulations/fixture_predictions.csv"):
        st.warning("⚠️ Please run simulations first.")
        return

    try:
        # Load fixture predictions with error handling
        try:
            predictions_df = pd.read_csv("reports/simulations/fixture_predictions.csv", on_bad_lines='skip')
        except pd.errors.ParserError as e:
            st.error(f"Error reading fixture predictions file: {e}")
            st.info("Trying to regenerate fixture predictions...")
            return

        # Load upcoming fixtures
        try:
            fixtures_df = pd.read_csv("data/clean/upcoming_fixtures.csv", parse_dates=['Date'])
        except Exception as e:
            st.error(f"Error reading upcoming fixtures: {e}")

            # Offer to clean the file
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔧 Clean Upcoming Fixtures File"):
                    with st.spinner("Cleaning CSV file..."):
                        if clean_upcoming_fixtures_file():
                            st.success("✅ File cleaned successfully! Please refresh the page.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to clean the file.")

            with col2:
                if st.button("📄 Show File Status"):
                    if os.path.exists("data/clean/upcoming_fixtures.csv"):
                        with open("data/clean/upcoming_fixtures.csv", 'r') as f:
                            lines = f.readlines()
                        st.info(f"File exists with {len(lines)} lines")
                        if len(lines) > 140:
                            st.warning("File appears to have malformed data around line 137")
                    else:
                        st.error("File does not exist")
            return

        # Group by match
        fixture_summary = predictions_df.groupby(['home_team', 'away_team']).agg({
            'home_win': 'mean',
            'draw': 'mean',
            'away_win': 'mean',
            'home_goals': 'mean',
            'away_goals': 'mean'
        }).reset_index()

        # Merge with fixture dates using correct column names
        fixture_summary = fixture_summary.merge(
            fixtures_df[['Home_Team', 'Away_Team', 'Date']],
            left_on=['home_team', 'away_team'],
            right_on=['Home_Team', 'Away_Team'],
            how='left'
        )

        # Sort by date
        fixture_summary = fixture_summary.sort_values('Date')

        # Display fixtures by date
        st.subheader("📅 Upcoming Fixtures with Predictions")

        # Filter controls
        col1, col2 = st.columns(2)

        with col1:
            # Date filter
            unique_dates = fixture_summary['Date'].dt.date.unique()
            selected_date = st.selectbox("Select Date", options=['All'] + list(unique_dates))

        with col2:
            # Team filter
            all_teams = sorted(set(fixture_summary['home_team'].unique()) | set(fixture_summary['away_team'].unique()))
            selected_team = st.selectbox("Select Team", options=['All Teams'] + all_teams)

        # Apply filters
        display_fixtures = fixture_summary.copy()

        if selected_date != 'All':
            display_fixtures = display_fixtures[display_fixtures['Date'].dt.date == selected_date]

        if selected_team != 'All Teams':
            # Filter for matches where the selected team is either home or away
            team_filter = (display_fixtures['home_team'] == selected_team) | (display_fixtures['away_team'] == selected_team)
            display_fixtures = display_fixtures[team_filter]

            # Show team-specific info
            if len(display_fixtures) > 0:
                home_games = len(display_fixtures[display_fixtures['home_team'] == selected_team])
                away_games = len(display_fixtures[display_fixtures['away_team'] == selected_team])
                st.info(f"📊 {selected_team}: {len(display_fixtures)} total games ({home_games} home, {away_games} away)")
            else:
                st.info(f"ℹ️ No upcoming fixtures found for {selected_team}")

        # Show filter results
        if len(display_fixtures) == 0:
            st.warning("No fixtures match the selected filters.")
            return
        elif len(display_fixtures) != len(fixture_summary):
            st.success(f"📋 Showing {len(display_fixtures)} of {len(fixture_summary)} total fixtures")

        # Display each fixture
        for _, match in display_fixtures.iterrows():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader(f"{match['home_team']} vs {match['away_team']}")
                if pd.notna(match['Date']):
                    st.caption(f"📅 {match['Date'].strftime('%Y-%m-%d')}")

                # Probability bars
                st.write("**Match Probabilities:**")
                col_h, col_d, col_a = st.columns(3)

                with col_h:
                    st.metric("Home Win", f"{match['home_win']:.1%}")
                with col_d:
                    st.metric("Draw", f"{match['draw']:.1%}")
                with col_a:
                    st.metric("Away Win", f"{match['away_win']:.1%}")

            with col2:
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

        # Option to regenerate predictions
        if st.button("🔄 Regenerate Fixture Predictions"):
            try:
                # Clean up the malformed file
                if os.path.exists("reports/simulations/fixture_predictions.csv"):
                    os.remove("reports/simulations/fixture_predictions.csv")
                st.success("Cleared malformed predictions file. Please run simulations again.")
                st.rerun()
            except Exception as cleanup_error:
                st.error(f"Error cleaning up file: {cleanup_error}")

def clean_upcoming_fixtures_file():
    """Clean up malformed upcoming fixtures CSV"""
    try:
        input_file = "data/clean/upcoming_fixtures.csv"
        if not os.path.exists(input_file):
            return False

        # Read the file line by line to handle malformed lines
        cleaned_lines = []
        expected_columns = ['Round', 'Date', 'Day', 'Time', 'Home_Team', 'Away_Team', 'Status']

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Keep header
        if lines:
            cleaned_lines.append(lines[0].strip())

            # Process each data line
            for i, line in enumerate(lines[1:], start=2):
                parts = line.strip().split(',')

                # If line has exactly 7 fields (expected), keep it
                if len(parts) == 7:
                    cleaned_lines.append(line.strip())
                # If line has more than 7 fields, truncate to first 7
                elif len(parts) > 7:
                    fixed_line = ','.join(parts[:7])
                    cleaned_lines.append(fixed_line)
                    print(f"Fixed line {i}: truncated from {len(parts)} to 7 fields")
                # If line has fewer than 7 fields, skip it
                else:
                    print(f"Skipped line {i}: only {len(parts)} fields")

        # Write cleaned file
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))

        print(f"Cleaned upcoming fixtures file: {len(cleaned_lines)-1} data rows")
        return True

    except Exception as e:
        print(f"Error cleaning upcoming fixtures: {e}")
        return False

def clean_fixture_predictions_file():
    """Clean up malformed fixture predictions CSV"""
    try:
        if os.path.exists("reports/simulations/fixture_predictions.csv"):
            # Read with error handling
            df = pd.read_csv("reports/simulations/fixture_predictions.csv", on_bad_lines='skip')

            # Clean team names (remove commas)
            if 'home_team' in df.columns:
                df['home_team'] = df['home_team'].astype(str).str.replace(',', '')
            if 'away_team' in df.columns:
                df['away_team'] = df['away_team'].astype(str).str.replace(',', '')
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str).str.replace(',', '')

            # Save cleaned file
            df.to_csv("reports/simulations/fixture_predictions.csv", index=False)
            return True
    except Exception as e:
        print(f"Error cleaning fixture predictions: {e}")
        return False

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

def odds_integration_page():
    st.header("🎯 Odds Integration & Hybrid Predictions")
    st.markdown("### Combine Statistical Models with Real Betting Markets")

    # Initialize session state for odds
    if 'odds_data' not in st.session_state:
        st.session_state.odds_data = OddsData()
    if 'odds_fetched' not in st.session_state:
        st.session_state.odds_fetched = False

    # Check if we have a trained model
    if not st.session_state.get('model_trained', False):
        # Try to load existing model from disk
        if os.path.exists("models/poisson_params.pkl"):
            try:
                from src.models.poisson_model import PoissonModel
                model = PoissonModel()
                model.load("models/poisson_params.pkl")
                st.session_state.poisson_model = model
                st.session_state.model_trained = True
                st.info("📄 Loaded existing trained model from disk")
            except Exception as e:
                st.warning("⚠️ Please train the Poisson model first in the 'Model Training' section")
                st.error(f"Failed to load existing model: {str(e)}")
                return
        else:
            st.warning("⚠️ Please train the Poisson model first in the 'Model Training' section")
            return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📡 Live Odds Data")

        # API Status
        try:
            odds_api = OddsAPI()
            st.success("✅ Odds API connected and ready")

            # Check API usage
            if st.button("Check API Usage", type="secondary"):
                with st.spinner("Checking API usage..."):
                    usage = odds_api.check_api_usage()
                    if usage:
                        st.info(f"📊 API Usage: {usage.get('requests_used', 0)} used, {usage.get('requests_remaining', 0)} remaining")
                    else:
                        st.warning("Could not retrieve usage information")

        except Exception as e:
            st.error(f"❌ Odds API connection failed: {str(e)}")
            st.info("🔑 Make sure your ODDS_API_KEY is properly configured")
            return

        # Fetch current odds
        if st.button("Fetch Current Odds", type="primary"):
            with st.spinner("Fetching live odds from bookmakers..."):
                try:
                    odds_records = odds_api.get_upcoming_matches_odds()

                    if odds_records:
                        # Store in session state
                        st.session_state.odds_data.clear_cache()
                        for record in odds_records:
                            st.session_state.odds_data.add_match_odds(record)

                        st.session_state.odds_fetched = True
                        st.success(f"✅ Successfully fetched odds for {len(odds_records)} matches!")

                        # Display summary
                        st.write("**Fetched Matches:**")
                        for record in odds_records[:5]:  # Show first 5
                            margin = calculate_margin(record.home_odds, record.draw_odds, record.away_odds)
                            st.write(f"• {record.home_team} vs {record.away_team} - Margin: {margin:.1f}%")

                        if len(odds_records) > 5:
                            st.write(f"... and {len(odds_records) - 5} more matches")

                    else:
                        st.warning("No upcoming matches found with odds")

                except Exception as e:
                    st.error(f"Error fetching odds: {str(e)}")

    with col2:
        st.subheader("⚙️ Configuration")

        # Odds weight configuration
        st.write("**Odds Weight by Season Progress:**")
        early_weight = st.slider("Early Season (Games 1-2)", 0.0, 1.0, 0.7, 0.1)
        mid_early_weight = st.slider("Mid-Early (Games 3-5)", 0.0, 1.0, 0.5, 0.1)
        mid_weight = st.slider("Mid Season (Games 6-10)", 0.0, 1.0, 0.3, 0.1)
        late_weight = st.slider("Late Season (Games 11+)", 0.0, 1.0, 0.1, 0.1)

        # Value betting threshold
        st.write("**Value Betting:**")
        min_edge = st.slider("Minimum Edge (%)", 0.0, 20.0, 5.0, 0.5) / 100

    # Show current odds if available
    if st.session_state.odds_fetched and st.session_state.odds_data.get_all_odds():
        st.subheader("📊 Current Odds Analysis")

        odds_data = st.session_state.odds_data.get_all_odds()

        # Create odds DataFrame for display
        odds_list = []
        for key, record in odds_data.items():
            margin = calculate_margin(record.home_odds, record.draw_odds, record.away_odds)
            home_prob, draw_prob, away_prob = remove_margin(record.home_odds, record.draw_odds, record.away_odds)

            odds_list.append({
                'Match': f"{record.home_team} vs {record.away_team}",
                'Date': record.date.strftime('%Y-%m-%d %H:%M'),
                'Home Odds': f"{record.home_odds:.2f}",
                'Draw Odds': f"{record.draw_odds:.2f}",
                'Away Odds': f"{record.away_odds:.2f}",
                'Margin %': f"{margin:.1f}%",
                'Home Prob': f"{home_prob:.1%}",
                'Draw Prob': f"{draw_prob:.1%}",
                'Away Prob': f"{away_prob:.1%}"
            })

        if odds_list:
            odds_df = pd.DataFrame(odds_list)
            st.dataframe(odds_df, use_container_width=True)

    # Hybrid predictions section
    if st.session_state.odds_fetched and st.session_state.get('model_trained', False):
        st.subheader("🔮 Hybrid Predictions")
        st.markdown("*Combining Poisson Model + Betting Odds*")

        # Load the trained model
        try:
            model = st.session_state.get('poisson_model')
            if model is None:
                st.warning("Model not found in session. Please retrain the model.")
                return

            # Create hybrid model
            hybrid_model = HybridPoissonOddsModel(model)

            # Get season progress estimate
            season_progress = st.slider("Season Progress", 0.0, 1.0, 0.5, 0.05, 
                                       help="0.0 = Season start, 1.0 = Season end")

            # Generate predictions for all matches with odds
            predictions = []
            value_bets = []

            for key, odds_record in st.session_state.odds_data.get_all_odds().items():
                try:
                    # Get hybrid prediction
                    prediction = hybrid_model.predict_match_detailed(
                        odds_record.home_team,
                        odds_record.away_team,
                        odds_record,
                        season_progress
                    )

                    predictions.append(prediction)

                    # Check for value bets
                    if prediction.get('has_odds', False):
                        values = hybrid_model.calculate_value_bets(
                            prediction['combined_probs'],
                            odds_record,
                            min_edge
                        )
                        if values:
                            value_bets.extend([{**v, 'match': f"{odds_record.home_team} vs {odds_record.away_team}"} for v in values])

                except Exception as e:
                    st.error(f"Error predicting {odds_record.home_team} vs {odds_record.away_team}: {str(e)}")
                    continue

            if predictions:
                # Display predictions
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**🎯 Match Predictions**")

                    pred_data = []
                    for pred in predictions[:10]:  # Show first 10
                        if pred.get('has_odds', False):
                            poisson_winner = ['Home', 'Draw', 'Away'][np.argmax(pred['poisson_probs'])]
                            combined_winner = ['Home', 'Draw', 'Away'][np.argmax(pred['combined_probs'])]

                            pred_data.append({
                                'Match': f"{pred['home_team']} vs {pred['away_team']}",
                                'Poisson Winner': poisson_winner,
                                'Combined Winner': combined_winner,
                                'Odds Weight': f"{pred['odds_weight']:.1%}",
                                'Agreement': '✅' if poisson_winner == combined_winner else '❌'
                            })

                    if pred_data:
                        pred_df = pd.DataFrame(pred_data)
                        st.dataframe(pred_df, use_container_width=True)

                with col2:
                    st.write("**💰 Value Betting Opportunities**")

                    if value_bets:
                        # Sort by edge
                        value_bets.sort(key=lambda x: x['edge'], reverse=True)

                        for bet in value_bets[:5]:  # Show top 5
                            st.write(f"**{bet['match']}**")
                            st.write(f"• Bet: {bet['outcome'].title()}")
                            st.write(f"• Our Probability: {bet['our_probability']:.1%}")
                            st.write(f"• Market Odds: {bet['market_odds']:.2f}")
                            st.write(f"• Edge: {bet['edge_percent']:+.1f}%")
                            st.write("---")
                    else:
                        st.info("No value bets found with current threshold")

            # Model comparison visualization
            if len(predictions) >= 3:
                st.subheader("📈 Model Comparison")

                # Create comparison chart
                comparison_data = []
                for pred in predictions:
                    if pred.get('has_odds', False):
                        comparison_data.append({
                            'Match': f"{pred['home_team'][:3]} vs {pred['away_team'][:3]}",
                            'Poisson Home': pred['poisson_probs'][0],
                            'Odds Home': pred['odds_probs'][0],
                            'Combined Home': pred['combined_probs'][0]
                        })

                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=comp_df['Match'], y=comp_df['Poisson Home'], 
                                           mode='markers+lines', name='Poisson Model', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=comp_df['Match'], y=comp_df['Odds Home'], 
                                           mode='markers+lines', name='Betting Odds', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=comp_df['Match'], y=comp_df['Combined Home'], 
                                           mode='markers+lines', name='Hybrid Model', line=dict(color='green')))

                    fig.update_layout(title="Home Win Probability Comparison", 
                                    xaxis_title="Match", yaxis_title="Probability",
                                    height=400)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating hybrid predictions: {str(e)}")

    # Help section
    with st.expander("ℹ️ How Odds Integration Works"):
        st.markdown("""
        **Hybrid Model Approach:**

        1. **Early Season**: Betting odds get higher weight (70%) because limited historical data
        2. **Mid Season**: Balanced weighting (30-50%) as patterns emerge
        3. **Late Season**: Statistical model dominates (90%) with extensive data

        **Value Betting:**
        - Compares our model's probabilities vs market odds
        - Identifies opportunities where market undervalues outcomes
        - Edge = (Our Probability × Odds) - 1.0

        **Data Sources:**
        - Statistical Model: Historical match results + team strengths
        - Betting Odds: Live markets from European bookmakers
        - Combination: Weighted average based on season progress
        """)

if __name__ == "__main__":
    main()