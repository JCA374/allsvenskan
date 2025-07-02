import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Import custom modules
from src.data.scraper import AllsvenskanScraper
from src.data.cleaner import DataCleaner
from src.data.strength import TeamStrengthCalculator
from src.models.poisson_model import PoissonModel
from src.simulation.simulator import MonteCarloSimulator
from src.analysis.aggregator import ResultsAggregator
from src.visualization.dashboard import Dashboard

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

def main():
    st.title("⚽ Allsvenskan Monte Carlo Forecast")
    st.markdown("### Predicting Swedish Football League Outcomes Using Statistical Modeling")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Collection", "Model Training", "Monte Carlo Simulation", "Results Analysis", "Dashboard"]
    )
    
    if page == "Data Collection":
        data_collection_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Monte Carlo Simulation":
        simulation_page()
    elif page == "Results Analysis":
        analysis_page()
    elif page == "Dashboard":
        dashboard_page()

def data_collection_page():
    st.header("📊 Data Collection & Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Web Scraping")
        if st.button("🔍 Scrape Latest Data from Allsvenskan.se", type="primary"):
            with st.spinner("Scraping data from allsvenskan.se..."):
                try:
                    scraper = AllsvenskanScraper()
                    raw_data = scraper.scrape_matches()
                    
                    if raw_data is not None and not raw_data.empty:
                        # Save raw data
                        os.makedirs("data/raw", exist_ok=True)
                        raw_data.to_csv("data/raw/fixtures_results_raw.csv", index=False)
                        st.success(f"✅ Successfully scraped {len(raw_data)} matches!")
                        st.dataframe(raw_data.head())
                        
                        # Clean data automatically
                        cleaner = DataCleaner()
                        results, fixtures = cleaner.clean_data(raw_data)
                        
                        os.makedirs("data/clean", exist_ok=True)
                        results.to_csv("data/clean/results.csv", index=False)
                        fixtures.to_csv("data/clean/fixtures.csv", index=False)
                        
                        st.session_state.data_loaded = True
                        st.success(f"✅ Data cleaned: {len(results)} completed matches, {len(fixtures)} upcoming fixtures")
                        
                    else:
                        st.error("❌ Failed to scrape data. Please check the website or try again later.")
                        
                except Exception as e:
                    st.error(f"❌ Error during scraping: {str(e)}")
    
    with col2:
        st.subheader("Data Status")
        
        # Check for existing data
        results_exist = os.path.exists("data/clean/results.csv")
        fixtures_exist = os.path.exists("data/clean/fixtures.csv")
        
        if results_exist and fixtures_exist:
            st.success("✅ Clean data files found")
            results = pd.read_csv("data/clean/results.csv")
            fixtures = pd.read_csv("data/clean/fixtures.csv")
            
            st.metric("Completed Matches", len(results))
            st.metric("Upcoming Fixtures", len(fixtures))
            
            st.session_state.data_loaded = True
            
            # Show recent results
            if len(results) > 0:
                st.subheader("Recent Results")
                recent = results.tail(5)
                for _, match in recent.iterrows():
                    st.write(f"{match['HomeTeam']} {match['FTHG']}-{match['FTAG']} {match['AwayTeam']}")
        else:
            st.warning("⚠️ No clean data found. Please scrape data first.")

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
                    results = pd.read_csv("data/clean/results.csv")
                    
                    strength_calc = TeamStrengthCalculator()
                    team_stats = strength_calc.calculate_strengths(results)
                    
                    os.makedirs("data/processed", exist_ok=True)
                    team_stats.to_csv("data/processed/team_stats.csv")
                    
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
            champ_df = pd.DataFrame(list(championship_odds.items()), 
                                  columns=['Team', 'Championship %'])
            champ_df['Championship %'] = champ_df['Championship %'] * 100
            champ_df = champ_df.sort_values('Championship %', ascending=False)
            
            fig_champ = px.bar(champ_df.head(8), x='Team', y='Championship %',
                              title="Top 8 Championship Contenders")
            fig_champ.update_xaxis(tickangle=45)
            st.plotly_chart(fig_champ, use_container_width=True)
            
        with col2:
            st.subheader("⬇️ Relegation Probabilities")
            releg_df = pd.DataFrame(list(relegation_odds.items()), 
                                  columns=['Team', 'Relegation %'])
            releg_df['Relegation %'] = releg_df['Relegation %'] * 100
            releg_df = releg_df.sort_values('Relegation %', ascending=False)
            
            fig_releg = px.bar(releg_df.head(8), x='Team', y='Relegation %',
                              title="Top 8 Relegation Candidates", color_discrete_sequence=['red'])
            fig_releg.update_xaxis(tickangle=45)
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
        points_df = pd.DataFrame(list(expected_points.items()), 
                               columns=['Team', 'Expected Points'])
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

if __name__ == "__main__":
    main()
