import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from allsvenskan.data.scraper import AllsvenskanScraper
from allsvenskan.data.cleaner import DataCleaner
from allsvenskan.data.strength import TeamStrengthCalculator
from allsvenskan.models.poisson_model import PoissonModel
from allsvenskan.simulation.simulator import MonteCarloSimulator
from allsvenskan.analysis.aggregator import ResultsAggregator

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Allsvenskan Forecast",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_PATH = Path("data/clean/results.csv")
FIXTURES_PATH = Path("data/clean/fixtures.csv")
HISTORICAL_PATH = Path("data/clean/historical_results.csv")
TEAM_STATS_PATH = Path("data/processed/team_stats.csv")
MODEL_PATH = Path("models/poisson_params.pkl")
SIM_PATH = Path("reports/simulations/sim_results_latest.csv")

RELEGATION_SPOTS = 3
EUROPEAN_SPOTS = 5   # top 5: CL (1-2) + EL/ECL (3-5)

# ── Session-state defaults ─────────────────────────────────────────────────────
for key, default in {
    "data_loaded": False,
    "model_trained": False,
    "sim_complete": False,
    "active_page": "Data",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Detect what's already on disk so refreshes don't lose context
if not st.session_state.data_loaded and RESULTS_PATH.exists():
    try:
        _r = pd.read_csv(RESULTS_PATH)
        if len(_r) > 0:
            st.session_state.data_loaded = True
    except Exception:
        pass

if not st.session_state.model_trained and MODEL_PATH.exists():
    st.session_state.model_trained = True

if not st.session_state.sim_complete and SIM_PATH.exists():
    try:
        _s = pd.read_csv(SIM_PATH)
        if len(_s) > 0:
            st.session_state.sim_complete = True
    except Exception:
        pass

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Allsvenskan")
    st.caption("Monte Carlo Forecast")
    st.divider()

    pages = ["Data", "Model", "Simulate", "Forecast", "Fixtures"]
    icons  = ["🗄️",   "🧠",    "🎲",       "📊",       "📅"]

    for page, icon in zip(pages, icons):
        label = f"{icon} {page}"
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.active_page == page else "secondary"):
            st.session_state.active_page = page
            st.rerun()

    st.divider()
    # Pipeline status
    st.caption("Pipeline status")
    st.write("Data" , "✅" if st.session_state.data_loaded   else "⬜")
    st.write("Model", "✅" if st.session_state.model_trained  else "⬜")
    st.write("Sim"  , "✅" if st.session_state.sim_complete   else "⬜")

page = st.session_state.active_page


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
    return df[df["FTHG"].notna() & df["FTAG"].notna()].copy()

def _load_fixtures() -> pd.DataFrame:
    if FIXTURES_PATH.exists():
        df = pd.read_csv(FIXTURES_PATH, parse_dates=["Date"])
        if "FTHG" in df.columns:
            df = df[df["FTHG"].isna()]
        return df[["Date", "HomeTeam", "AwayTeam"]].copy()
    return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])

def _load_model() -> PoissonModel:
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, PoissonModel):
        return data
    # Legacy: file contains the raw dict saved by model.save()
    m = PoissonModel()
    m.load(str(MODEL_PATH))
    return m

def _load_sim() -> pd.DataFrame:
    return pd.read_csv(SIM_PATH)

def _standings_from_results(results: pd.DataFrame) -> pd.DataFrame:
    teams = pd.unique(results[["HomeTeam", "AwayTeam"]].values.ravel())
    cols = ["GP", "W", "D", "L", "GF", "GA", "GD", "Pts"]
    tbl = pd.DataFrame(0, index=teams, columns=cols)
    for _, r in results.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        tbl.at[h, "GP"] += 1; tbl.at[a, "GP"] += 1
        tbl.at[h, "GF"] += hg; tbl.at[h, "GA"] += ag
        tbl.at[a, "GF"] += ag; tbl.at[a, "GA"] += hg
        if hg > ag:
            tbl.at[h, "W"] += 1; tbl.at[a, "L"] += 1
            tbl.at[h, "Pts"] += 3
        elif ag > hg:
            tbl.at[a, "W"] += 1; tbl.at[h, "L"] += 1
            tbl.at[a, "Pts"] += 3
        else:
            tbl.at[h, "D"] += 1; tbl.at[a, "D"] += 1
            tbl.at[h, "Pts"] += 1; tbl.at[a, "Pts"] += 1
    tbl["GD"] = tbl["GF"] - tbl["GA"]
    return (
        tbl.sort_values(["Pts", "GD", "GF"], ascending=False)
           .reset_index()
           .rename(columns={"index": "Team"})
    )

def _color_table_row(row, n_teams):
    """Return CSS background colours for standing row based on position."""
    pos = row.name + 1  # 1-indexed
    if pos == 1:
        return ["background-color: #d4edda"] * len(row)  # gold-ish green
    if pos <= 2:
        return ["background-color: #e8f5e9"] * len(row)  # CL
    if pos <= EUROPEAN_SPOTS:
        return ["background-color: #e3f2fd"] * len(row)  # Europe
    if pos > n_teams - RELEGATION_SPOTS:
        return ["background-color: #fce4ec"] * len(row)  # relegation
    return [""] * len(row)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA
# ══════════════════════════════════════════════════════════════════════════════
if page == "Data":
    st.title("Data")

    col_load, col_status = st.columns([1, 2])

    with col_load:
        current_year = pd.Timestamp.now().year
        history_exists = HISTORICAL_PATH.exists() and len(pd.read_csv(HISTORICAL_PATH)) > 0 if HISTORICAL_PATH.exists() else False

        # ── One-time historical download ──────────────────────────────────────
        st.subheader("Historical data (one-time)")
        if history_exists:
            hist_df = pd.read_csv(HISTORICAL_PATH)
            hist_seasons = sorted(hist_df["SeasonStart"].dropna().astype(int).unique())
            st.success(f"Cached: seasons {hist_seasons[0]}–{hist_seasons[-1]} ({len(hist_df)} matches)")
        else:
            st.info("No historical data cached yet.")

        if st.button("Download All History", disabled=history_exists):
            past_years = list(range(2012, current_year))  # all completed seasons
            with st.spinner(f"Downloading seasons {past_years[0]}–{past_years[-1]}…"):
                try:
                    scraper = AllsvenskanScraper()
                    raw = scraper.scrape_matches(seasons=past_years)
                    if raw.empty:
                        st.error("No historical data returned.")
                    else:
                        cleaner = DataCleaner()
                        hist_results, _ = cleaner.clean_data(raw)
                        HISTORICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
                        hist_results.to_csv(HISTORICAL_PATH, index=False)
                        st.success(f"Saved {len(hist_results)} historical matches.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Download failed: {e}")

        st.divider()

        # ── Current season refresh ────────────────────────────────────────────
        st.subheader(f"Current season ({current_year})")

        if st.button("Refresh Current Season", type="primary"):
            with st.spinner(f"Fetching {current_year} data…"):
                try:
                    scraper = AllsvenskanScraper()
                    raw = scraper.scrape_matches(seasons=[current_year])
                    if raw.empty:
                        st.error(f"No data returned for {current_year}.")
                    else:
                        cleaner = DataCleaner()
                        cur_results, cur_fixtures = cleaner.clean_data(raw)

                        # Merge with historical (if available)
                        if history_exists:
                            hist_df = pd.read_csv(HISTORICAL_PATH, parse_dates=["Date"])
                            combined = pd.concat([hist_df, cur_results], ignore_index=True)
                            combined = combined.drop_duplicates(
                                subset=["Date", "HomeTeam", "AwayTeam"], keep="last"
                            )
                        else:
                            combined = cur_results

                        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                        FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
                        combined.to_csv(RESULTS_PATH, index=False)
                        cur_fixtures.to_csv(FIXTURES_PATH, index=False)
                        cur_fixtures.to_csv("data/clean/upcoming_fixtures.csv", index=False)

                        st.session_state.data_loaded = True
                        st.success(
                            f"Updated: {len(cur_results)} matches this season, "
                            f"{len(cur_fixtures)} upcoming fixtures. "
                            f"{len(combined)} total results saved."
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")

    with col_status:
        if st.session_state.data_loaded:
            try:
                results = _load_results()
                fixtures = _load_fixtures()

                c1, c2, c3 = st.columns(3)
                c1.metric("Results", len(results))
                c2.metric("Upcoming fixtures", len(fixtures))
                seasons = sorted(results["SeasonStart"].dropna().astype(int).unique()) if "SeasonStart" in results.columns else []
                c3.metric("Seasons loaded", len(seasons) if seasons else "—")

            except Exception as e:
                st.warning(f"Could not read data files: {e}")
        else:
            st.info("No data loaded yet. Select seasons and click Download.")

    if st.session_state.data_loaded:
        st.divider()
        st.subheader("Current Season Standings")
        try:
            results = _load_results()
            # Filter to most recent season
            if "SeasonStart" in results.columns:
                latest = results["SeasonStart"].max()
                results_current = results[results["SeasonStart"] == latest]
            else:
                results_current = results

            standings = _standings_from_results(results_current)
            n = len(standings)

            styled = (
                standings.style
                .apply(_color_table_row, n_teams=n, axis=1)
                .format({"GD": "{:+d}"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.caption(
                "🟢 Title contender  |  🔵 European spots  |  🔴 Relegation zone"
            )
        except Exception as e:
            st.error(f"Could not build standings: {e}")

        st.divider()
        st.subheader("Browse historical matches")
        try:
            all_results = _load_results()
            teams = sorted(pd.unique(all_results[["HomeTeam", "AwayTeam"]].values.ravel()))

            fc1, fc2, fc3 = st.columns(3)
            sel_home = fc1.selectbox("Home team", ["Any"] + teams, key="hist_home")
            sel_away = fc2.selectbox("Away team", ["Any"] + teams, key="hist_away")
            seasons_avail = sorted(all_results["SeasonStart"].dropna().astype(int).unique(), reverse=True) if "SeasonStart" in all_results.columns else []
            sel_season = fc3.selectbox("Season", ["All"] + [str(s) for s in seasons_avail], key="hist_season")

            view = all_results.copy()
            if sel_home != "Any":
                view = view[view["HomeTeam"] == sel_home]
            if sel_away != "Any":
                view = view[view["AwayTeam"] == sel_away]
            if sel_season != "All":
                view = view[view["SeasonStart"] == int(sel_season)]

            view = view.sort_values("Date", ascending=False)
            st.caption(f"{len(view)} matches")
            st.dataframe(
                view[["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam", "SeasonStart"]]
                .rename(columns={"FTHG": "HG", "FTAG": "AG", "SeasonStart": "Season"})
                .reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Could not load match history: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model":
    st.title("Model Training")

    if not st.session_state.data_loaded:
        st.warning("Load data first (Data page).")
        st.stop()

    col_train, col_result = st.columns([1, 2])

    with col_train:
        st.subheader("Train Poisson Model")
        mode = st.radio(
            "Training mode",
            ["Fast", "Advanced (MLE + Dixon-Coles)"],
            help="Advanced is more accurate but takes longer.",
        )
        advanced = mode.startswith("Advanced")

        if st.button("Train Model", type="primary"):
            with st.spinner("Calculating team strengths…"):
                try:
                    results = _load_results()
                    strength_calc = TeamStrengthCalculator(use_odds_integration=False)
                    team_stats = strength_calc.calculate_strengths(results)
                    TEAM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    team_stats.to_csv(TEAM_STATS_PATH)
                except Exception as e:
                    st.error(f"Strength calculation failed: {e}")
                    st.stop()

            with st.spinner(f"Training {'advanced' if advanced else 'fast'} Poisson model…"):
                try:
                    results = _load_results()
                    team_stats = pd.read_csv(TEAM_STATS_PATH, index_col=0)
                    model = PoissonModel(use_mle=advanced, use_dixon_coles=advanced)
                    model.fit(results, team_stats)
                    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    model.save(str(MODEL_PATH))
                    st.session_state.model_trained = True
                    st.success("Model trained and saved.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with col_result:
        if st.session_state.model_trained:
            try:
                model = _load_model()
                st.subheader("Model Parameters")
                mc1, mc2 = st.columns(2)
                mc1.metric("Home Advantage", f"{model.home_advantage:.3f}")
                mc2.metric("League Avg Goals/Match", f"{model.league_avg:.3f}")

                if TEAM_STATS_PATH.exists():
                    st.subheader("Team Strengths")
                    ts = pd.read_csv(TEAM_STATS_PATH, index_col=0)
                    display_cols = [c for c in ["attack_strength", "defense_strength", "avg_goals_scored", "avg_goals_conceded"] if c in ts.columns]
                    if display_cols:
                        ts_show = ts[display_cols].copy().sort_values("attack_strength", ascending=False)
                        ts_show.index.name = "Team"
                        ts_show.columns = [c.replace("_", " ").title() for c in ts_show.columns]

                        # Bar chart: attack vs defence
                        fig = go.Figure()
                        if "Attack Strength" in ts_show.columns:
                            fig.add_bar(x=ts_show.index, y=ts_show["Attack Strength"], name="Attack", marker_color="#2196F3")
                        if "Defense Strength" in ts_show.columns:
                            fig.add_bar(x=ts_show.index, y=ts_show["Defense Strength"], name="Defense", marker_color="#F44336")
                        fig.update_layout(
                            barmode="group", xaxis_tickangle=-45,
                            height=340, margin=dict(t=20, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(ts_show.round(3), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display model info: {e}")
        else:
            st.info("No model trained yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIMULATE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Simulate":
    st.title("Monte Carlo Simulation")

    if not st.session_state.model_trained:
        st.warning("Train the model first (Model page).")
        st.stop()

    col_cfg, col_info = st.columns([1, 2])

    with col_cfg:
        st.subheader("Settings")
        n_sims = st.slider(
            "Simulations",
            min_value=500, max_value=50_000, value=10_000, step=500,
            help="More simulations = more accurate probabilities, but slower.",
        )

        fixtures = _load_fixtures()
        upcoming_path = Path("data/clean/upcoming_fixtures.csv")
        if upcoming_path.exists():
            try:
                uf = pd.read_csv(upcoming_path)
                n_fixtures = len(uf)
            except Exception:
                n_fixtures = len(fixtures)
        else:
            n_fixtures = len(fixtures)

        st.metric("Upcoming fixtures", n_fixtures)

        if n_fixtures == 0:
            st.warning("No upcoming fixtures found. Re-download data for the current season.")

        run_disabled = n_fixtures == 0

        if st.button("Run Simulation", type="primary", disabled=run_disabled):
            progress = st.progress(0, text="Starting…")

            def _cb(pct):
                progress.progress(int(pct), text=f"{pct:.0f}%")

            try:
                model = _load_model()
                simulator = MonteCarloSimulator.from_upcoming_fixtures(model)

                # Seed simulations with already-accumulated points this season
                current_year = pd.Timestamp.now().year
                try:
                    _all_res = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
                    _cur = _all_res[_all_res["SeasonStart"] == current_year] if "SeasonStart" in _all_res.columns else pd.DataFrame()
                    if not _cur.empty:
                        _standings = _standings_from_results(_cur)
                        current_pts = _standings["Pts"].to_dict()
                    else:
                        current_pts = {}
                except Exception:
                    current_pts = {}

                if current_pts:
                    sim_results = simulator.run_monte_carlo_with_standings(
                        n_simulations=n_sims,
                        current_standings=current_pts,
                        progress_callback=_cb,
                    )
                else:
                    sim_results = simulator.run(n_simulations=n_sims, progress_callback=_cb)

                SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
                sim_results.to_csv(SIM_PATH, index=False)
                st.session_state.sim_complete = True
                progress.progress(100, text="Done!")
                st.success(f"Completed {n_sims:,} simulations over {n_fixtures} fixtures.")
                st.rerun()
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    with col_info:
        if st.session_state.sim_complete:
            try:
                sim = _load_sim()
                agg = ResultsAggregator()
                summary = agg.analyze_results(sim)
                if not summary.empty:
                    st.subheader("Quick Summary")
                    top3 = summary.head(3)
                    for _, row in top3.iterrows():
                        st.metric(row["Team"], f"{row['Mean_Points']:.1f} pts avg")
            except Exception as e:
                st.warning(f"Could not load previous results: {e}")
        else:
            st.info("No simulation run yet for the current data.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":
    st.title("Season Forecast")

    if not st.session_state.sim_complete:
        st.warning("Run a simulation first (Simulate page).")
        st.stop()

    try:
        sim = _load_sim()
        agg = ResultsAggregator()

        table = agg.generate_final_table_prediction(sim)
        champ  = agg.calculate_championship_odds(sim)
        releg  = agg.calculate_relegation_odds(sim, relegation_spots=RELEGATION_SPOTS)
        europe = agg.calculate_european_qualification_odds(sim, european_spots=EUROPEAN_SPOTS)
        pos_probs = agg.calculate_position_probabilities(sim)
        summary = agg.analyze_results(sim)
    except Exception as e:
        st.error(f"Could not load simulation results: {e}")
        st.stop()

    n_teams = len(table)

    # ── Expected Final Table ──────────────────────────────────────────────────
    st.subheader("Expected Final Standings")

    # Compute total games per team for the current season
    current_year = pd.Timestamp.now().year
    try:
        all_results = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
        season_results = (
            all_results[all_results["SeasonStart"] == current_year]
            if "SeasonStart" in all_results.columns
            else pd.DataFrame()
        )
    except Exception:
        season_results = pd.DataFrame()

    try:
        season_fixtures = pd.read_csv(FIXTURES_PATH, parse_dates=["Date"])
    except Exception:
        season_fixtures = pd.DataFrame()

    def _games_for_team(team, results_df, fixtures_df):
        played = 0
        if not results_df.empty:
            played = int(
                (results_df["HomeTeam"] == team).sum()
                + (results_df["AwayTeam"] == team).sum()
            )
        upcoming = 0
        if not fixtures_df.empty:
            upcoming = int(
                (fixtures_df["HomeTeam"] == team).sum()
                + (fixtures_df["AwayTeam"] == team).sum()
            )
        return played + upcoming

    # Build enriched table
    tbl = table.copy()
    tbl["GP"] = tbl["Team"].map(
        lambda t: _games_for_team(t, season_results, season_fixtures)
    )
    tbl["Title %"]    = tbl["Team"].map(lambda t: champ.get(t, 0) * 100)
    tbl["Europe %"]   = tbl["Team"].map(lambda t: europe.get(t, 0) * 100)
    tbl["Relegation %"] = tbl["Team"].map(lambda t: releg.get(t, 0) * 100)
    if not summary.empty:
        std_map = dict(zip(summary["Team"], summary["Std_Points"]))
        tbl["Pts ±"]  = tbl["Team"].map(lambda t: std_map.get(t, 0))

    # Reorder columns
    cols_order = ["Position", "Team", "GP", "Expected_Points"]
    if "Pts ±" in tbl.columns:
        cols_order.append("Pts ±")
    cols_order += ["Title %", "Europe %", "Relegation %"]
    tbl = tbl[cols_order]
    tbl = tbl.rename(columns={"Expected_Points": "Exp Pts"})
    tbl = tbl.reset_index(drop=True)

    def _row_color(row):
        pos = int(row["Position"])
        if pos == 1:
            return ["background-color: #fffde7"] * len(row)
        if pos <= 2:
            return ["background-color: #e8f5e9"] * len(row)
        if pos <= EUROPEAN_SPOTS:
            return ["background-color: #e3f2fd"] * len(row)
        if pos > n_teams - RELEGATION_SPOTS:
            return ["background-color: #fce4ec"] * len(row)
        return [""] * len(row)

    fmt = {
        "Exp Pts": "{:.1f}",
        "Title %": "{:.1f}%",
        "Europe %": "{:.1f}%",
        "Relegation %": "{:.1f}%",
    }
    if "Pts ±" in tbl.columns:
        fmt["Pts ±"] = "±{:.1f}"

    styled = tbl.style.apply(_row_color, axis=1).format(fmt)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    legend_cols = st.columns(4)
    legend_cols[0].caption("🟡 Title contender")
    legend_cols[1].caption("🟢 Champions League")
    legend_cols[2].caption("🔵 European spots")
    legend_cols[3].caption("🔴 Relegation zone")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    tab_title, tab_europe, tab_releg, tab_heat = st.tabs(
        ["Title Race", "European Qualification", "Relegation Battle", "Position Heatmap"]
    )

    def _sorted_bar(probs: dict, color: str, title: str, threshold: float = 0.005):
        data = {k: v * 100 for k, v in probs.items() if v > threshold}
        df = pd.DataFrame({"Team": list(data.keys()), "Probability": list(data.values())})
        df = df.sort_values("Probability", ascending=True)
        fig = px.bar(
            df, x="Probability", y="Team", orientation="h",
            text=df["Probability"].map(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=[color],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="Probability (%)", yaxis_title="",
            margin=dict(l=10, r=20, t=20, b=10), height=max(300, len(df) * 32),
            showlegend=False,
        )
        return fig

    with tab_title:
        st.plotly_chart(_sorted_bar(champ, "#FFC107", "Title"), use_container_width=True)

    with tab_europe:
        st.plotly_chart(_sorted_bar(europe, "#2196F3", "European"), use_container_width=True)

    with tab_releg:
        st.plotly_chart(_sorted_bar(releg, "#F44336", "Relegation"), use_container_width=True)

    with tab_heat:
        if pos_probs:
            teams_ordered = tbl["Team"].tolist()
            n_pos = len(teams_ordered)
            matrix = np.array([
                [pos_probs.get(team, [0] * n_pos)[p] for p in range(n_pos)]
                for team in teams_ordered
            ])
            fig = go.Figure(go.Heatmap(
                z=matrix * 100,
                x=[f"#{p+1}" for p in range(n_pos)],
                y=teams_ordered,
                colorscale="Blues",
                text=np.round(matrix * 100, 1),
                texttemplate="%{text:.0f}%",
                textfont={"size": 9},
                showscale=True,
                colorbar=dict(title="Prob %"),
            ))
            fig.update_layout(
                xaxis_title="Final Position",
                yaxis_title="",
                height=max(400, n_pos * 30),
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Position probabilities not available.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FIXTURES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Fixtures":
    st.title("Fixture Predictions")

    if not st.session_state.model_trained:
        st.warning("Train the model first (Model page).")
        st.stop()

    fixtures = _load_fixtures()
    if fixtures.empty:
        st.info("No upcoming fixtures found. Download data for the current season.")
        st.stop()

    try:
        model = _load_model()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    # ── Filters ───────────────────────────────────────────────────────────────
    all_teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        team_filter = st.selectbox("Filter by team", ["All teams"] + all_teams)
    with col_f2:
        if "Date" in fixtures.columns and fixtures["Date"].notna().any():
            dates = sorted(fixtures["Date"].dt.date.dropna().unique())
            date_filter = st.selectbox("Filter by date", ["All dates"] + [str(d) for d in dates])
        else:
            date_filter = "All dates"

    # Apply filters
    disp = fixtures.copy()
    if team_filter != "All teams":
        disp = disp[(disp["HomeTeam"] == team_filter) | (disp["AwayTeam"] == team_filter)]
    if date_filter != "All dates":
        disp = disp[disp["Date"].dt.date.astype(str) == date_filter]

    st.caption(f"Showing {len(disp)} fixture{'s' if len(disp) != 1 else ''}")

    if disp.empty:
        st.info("No fixtures match the selected filters.")
        st.stop()

    # ── Build predictions table ───────────────────────────────────────────────
    rows = []
    for _, fix in disp.iterrows():
        try:
            pred = model.predict_outcome_probabilities(fix["HomeTeam"], fix["AwayTeam"])
            rows.append({
                "Date":     fix["Date"].date() if pd.notna(fix.get("Date")) else "—",
                "Home":     fix["HomeTeam"],
                "Away":     fix["AwayTeam"],
                "Home Win": pred["home_win"],
                "Draw":     pred["draw"],
                "Away Win": pred["away_win"],
                "xG Home":  pred["mu_home"],
                "xG Away":  pred["mu_away"],
            })
        except Exception:
            pass

    if not rows:
        st.info("No predictions could be generated.")
        st.stop()

    pred_df = pd.DataFrame(rows)

    # ── Compact table view ────────────────────────────────────────────────────
    st.subheader("All Fixtures")
    table_df = pred_df.copy()
    table_df["Home Win"] = table_df["Home Win"].map(lambda x: f"{x:.0%}")
    table_df["Draw"]     = table_df["Draw"].map(lambda x: f"{x:.0%}")
    table_df["Away Win"] = table_df["Away Win"].map(lambda x: f"{x:.0%}")
    table_df["xG Home"]  = table_df["xG Home"].map(lambda x: f"{x:.2f}")
    table_df["xG Away"]  = table_df["xG Away"].map(lambda x: f"{x:.2f}")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Detailed fixture cards ────────────────────────────────────────────────
    st.subheader("Match Details")
    for _, r in pred_df.iterrows():
        with st.container():
            left, mid, right = st.columns([3, 2, 3])

            home_w = float(str(r["Home Win"]).strip("%")) if isinstance(r["Home Win"], str) else r["Home Win"] * 100
            draw_w = float(str(r["Draw"]).strip("%")) if isinstance(r["Draw"], str) else r["Draw"] * 100
            away_w = float(str(r["Away Win"]).strip("%")) if isinstance(r["Away Win"], str) else r["Away Win"] * 100

            # Use raw numeric values
            hw = r["Home Win"] if isinstance(r["Home Win"], float) else home_w / 100
            dw = r["Draw"]     if isinstance(r["Draw"], float)     else draw_w / 100
            aw = r["Away Win"] if isinstance(r["Away Win"], float)  else away_w / 100
            xgh = r["xG Home"] if isinstance(r["xG Home"], float) else float(r["xG Home"])
            xga = r["xG Away"] if isinstance(r["xG Away"], float) else float(r["xG Away"])

            with left:
                st.markdown(f"**{r['Home']}**")
                st.caption(f"xG {xgh:.2f}")

            with mid:
                st.markdown(f"<div style='text-align:center; font-size:0.8em; color:#888'>{r['Date']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center; font-size:1.1em'>{hw:.0%} · {dw:.0%} · {aw:.0%}</div>", unsafe_allow_html=True)

                # Stacked probability bar
                fig = go.Figure(go.Bar(
                    x=[hw * 100], y=[""], orientation="h",
                    marker_color="#2196F3", name="Home", showlegend=False,
                ))
                fig.add_bar(x=[dw * 100], y=[""], orientation="h",
                            marker_color="#9E9E9E", name="Draw", showlegend=False)
                fig.add_bar(x=[aw * 100], y=[""], orientation="h",
                            marker_color="#F44336", name="Away", showlegend=False)
                fig.update_layout(
                    barmode="stack", height=40,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showticklabels=False, range=[0, 100]),
                    yaxis=dict(showticklabels=False),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with right:
                st.markdown(f"**{r['Away']}**")
                st.caption(f"xG {xga:.2f}")

            st.divider()
