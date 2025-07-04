import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class Dashboard:
    def __init__(self):
        pass
    
    def create_full_dashboard(self):
        """Create complete interactive dashboard"""
        try:
            # Load simulation results
            if not os.path.exists("reports/simulations/sim_results.csv"):
                st.error("❌ No simulation results found. Please run simulations first.")
                return
            
            sim_results = pd.read_csv("reports/simulations/sim_results.csv")
            
            from src.analysis.aggregator import ResultsAggregator
            aggregator = ResultsAggregator()
            
            # Load current standings
            from src.utils.helpers import calculate_current_standings_from_url
            current_standings_df = calculate_current_standings_from_url()
            
            # Generate all analyses
            analysis = aggregator.analyze_results(sim_results)
            position_probs = aggregator.calculate_position_probabilities(sim_results)
            championship_odds = aggregator.calculate_championship_odds(sim_results)
            relegation_odds = aggregator.calculate_relegation_odds(sim_results)
            european_odds = aggregator.calculate_european_qualification_odds(sim_results)
            final_table = aggregator.generate_final_table_prediction(sim_results)
            
            # Create dashboard sections
            self._create_overview_section(final_table, sim_results, current_standings_df)
            self._create_position_analysis(position_probs, championship_odds, relegation_odds, european_odds)
            self._create_team_comparison(analysis, sim_results)
            self._create_probability_charts(championship_odds, relegation_odds, european_odds)
            
        except Exception as e:
            st.error(f"❌ Error creating dashboard: {str(e)}")
    
    def _create_overview_section(self, final_table, sim_results, current_standings_df):
        """Create overview section with key metrics"""
        st.subheader("🏆 Season Outlook")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not final_table.empty:
                most_likely_champion = final_table.iloc[0]['Team']
                champion_prob = final_table.iloc[0]['Championship_Prob']
                st.metric("Most Likely Champion", most_likely_champion, f"{champion_prob}%")
        
        with col2:
            n_simulations = len(sim_results)
            st.metric("Simulations Run", f"{n_simulations:,}")
        
        with col3:
            if not final_table.empty:
                avg_points = final_table['Expected_Points'].mean()
                st.metric("Avg Expected Points", f"{avg_points:.1f}")
        
        with col4:
            teams_analyzed = len(sim_results.columns)
            st.metric("Teams Analyzed", teams_analyzed)
        
        # Display final table prediction
        st.subheader("📊 Predicted Final Table")
        if not final_table.empty:
            # Format the table for better display
            display_table = final_table.copy()
            
            # Add current standings information
            if not current_standings_df.empty:
                # Create current position mapping
                current_pos_map = dict(zip(current_standings_df['Team'], range(1, len(current_standings_df) + 1)))
                current_pts_map = dict(zip(current_standings_df['Team'], current_standings_df['Pts']))
                
                display_table['Current_Position'] = display_table['Team'].map(current_pos_map)
                display_table['Current_Points'] = display_table['Team'].map(current_pts_map)
                display_table['Position_Change'] = display_table['Current_Position'] - display_table['Position']
            else:
                display_table['Current_Position'] = "N/A"
                display_table['Current_Points'] = 0
                display_table['Position_Change'] = 0
            
            display_table['Championship_Prob'] = display_table['Championship_Prob'].apply(lambda x: f"{x}%")
            display_table['Relegation_Prob'] = display_table['Relegation_Prob'].apply(lambda x: f"{x}%")
            
            st.dataframe(
                display_table,
                column_config={
                    "Position": st.column_config.NumberColumn("Final Pos"),
                    "Team": st.column_config.TextColumn("Team"),
                    "Current_Position": st.column_config.NumberColumn("Current Pos"),
                    "Current_Points": st.column_config.NumberColumn("Current Pts"),
                    "Expected_Points": st.column_config.NumberColumn("Final Pts", format="%.1f"),
                    "Position_Change": st.column_config.NumberColumn("Pos Change", format="%+d"),
                    "Championship_Prob": st.column_config.TextColumn("Title %"),
                    "Relegation_Prob": st.column_config.TextColumn("Relegation %")
                },
                hide_index=True
            )
    
    def _create_position_analysis(self, position_probs, championship_odds, relegation_odds, european_odds):
        """Create position probability analysis"""
        st.subheader("📈 Position Probability Analysis")
        
        if not position_probs:
            st.warning("No position probability data available")
            return
        
        # Create position probability heatmap
        pos_df = pd.DataFrame(position_probs).T
        pos_df.columns = [f"Pos {i+1}" for i in range(len(pos_df.columns))]
        
        # Select teams for heatmap (top and bottom teams for readability)
        if len(pos_df) > 10:
            # Show top 5 and bottom 5 teams based on expected performance
            expected_points = {team: sum(prob * (len(pos_df) - pos) for pos, prob in enumerate(probs)) 
                             for team, probs in position_probs.items()}
            sorted_teams = sorted(expected_points.items(), key=lambda x: x[1], reverse=True)
            selected_teams = [team for team, _ in sorted_teams[:5]] + [team for team, _ in sorted_teams[-5:]]
            pos_df_display = pos_df.loc[selected_teams]
        else:
            pos_df_display = pos_df
        
        fig_heatmap = px.imshow(
            pos_df_display.values,
            x=pos_df_display.columns,
            y=pos_df_display.index,
            color_continuous_scale='RdYlBu_r',
            title="Final Position Probabilities (Selected Teams)",
            labels={'x': 'Final Position', 'y': 'Team', 'color': 'Probability'}
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def _create_team_comparison(self, analysis, sim_results):
        """Create team comparison section"""
        st.subheader("⚖️ Team Comparison")
        
        if analysis.empty:
            st.warning("No team analysis data available")
            return
        
        # Team selector
        teams = analysis['Team'].tolist()
        selected_teams = st.multiselect(
            "Select teams to compare:",
            teams,
            default=teams[:5] if len(teams) >= 5 else teams
        )
        
        if not selected_teams:
            st.info("Please select teams to compare")
            return
        
        # Filter data for selected teams
        selected_analysis = analysis[analysis['Team'].isin(selected_teams)]
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Expected points comparison
            fig_points = px.bar(
                selected_analysis,
                x='Team',
                y='Mean_Points',
                title="Expected Final Points",
                labels={'Mean_Points': 'Expected Points', 'Team': 'Team'}
            )
            fig_points.update_xaxes(tickangle=45)
            st.plotly_chart(fig_points, use_container_width=True)
        
        with col2:
            # Points range (min-max)
            fig_range = go.Figure()
            
            for _, team_data in selected_analysis.iterrows():
                fig_range.add_trace(go.Scatter(
                    x=[team_data['Team'], team_data['Team']],
                    y=[team_data['Min_Points'], team_data['Max_Points']],
                    mode='lines+markers',
                    name=team_data['Team'],
                    line=dict(width=6),
                    marker=dict(size=8)
                ))
            
            fig_range.update_layout(
                title="Points Range (Min-Max)",
                xaxis_title="Team",
                yaxis_title="Points",
                showlegend=False
            )
            st.plotly_chart(fig_range, use_container_width=True)
        
        # Points distribution for selected teams
        if len(selected_teams) <= 5:
            st.subheader("Points Distribution")
            
            fig_dist = go.Figure()
            
            for team in selected_teams:
                if team in sim_results.columns:
                    team_points = sim_results[team]
                    fig_dist.add_trace(go.Histogram(
                        x=team_points,
                        name=team,
                        opacity=0.7,
                        nbinsx=20
                    ))
            
            fig_dist.update_layout(
                title="Points Distribution by Team",
                xaxis_title="Final Points",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    def _create_probability_charts(self, championship_odds, relegation_odds, european_odds):
        """Create probability visualization charts"""
        st.subheader("🎯 Key Probabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if championship_odds:
                # Championship probabilities
                champ_data = [(team, prob * 100) for team, prob in championship_odds.items()]
                champ_data.sort(key=lambda x: x[1], reverse=True)
                champ_df = pd.DataFrame(champ_data[:8], columns=['Team', 'Probability'])
                
                fig_champ = px.bar(
                    champ_df,
                    x='Probability',
                    y='Team',
                    orientation='h',
                    title="Championship Probability (%)",
                    color='Probability',
                    color_continuous_scale='greens'
                )
                fig_champ.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_champ, use_container_width=True)
        
        with col2:
            if european_odds:
                # European qualification probabilities
                euro_data = [(team, prob * 100) for team, prob in european_odds.items()]
                euro_data.sort(key=lambda x: x[1], reverse=True)
                euro_df = pd.DataFrame(euro_data[:8], columns=['Team', 'Probability'])
                
                fig_euro = px.bar(
                    euro_df,
                    x='Probability',
                    y='Team',
                    orientation='h',
                    title="European Qualification (%)",
                    color='Probability',
                    color_continuous_scale='blues'
                )
                fig_euro.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_euro, use_container_width=True)
        
        with col3:
            if relegation_odds:
                # Relegation probabilities
                releg_data = [(team, prob * 100) for team, prob in relegation_odds.items()]
                releg_data.sort(key=lambda x: x[1], reverse=True)
                releg_df = pd.DataFrame(releg_data[:8], columns=['Team', 'Probability'])
                
                fig_releg = px.bar(
                    releg_df,
                    x='Probability',
                    y='Team',
                    orientation='h',
                    title="Relegation Probability (%)",
                    color='Probability',
                    color_continuous_scale='reds'
                )
                fig_releg.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_releg, use_container_width=True)
