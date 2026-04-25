# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Allsvenskan (Swedish top-flight football) Monte Carlo simulation and forecasting application. Predicts football match outcomes and season standings using Poisson-based statistical models, with optional live betting odds integration via a hybrid model. Has both a Streamlit web UI (`app.py`) and a CLI (`cli.py`).

## Development Commands

### Running the Application

```bash
# Streamlit UI
streamlit run app.py --server.port 5000

# CLI (full pipeline or individual steps)
python cli.py pipeline
python cli.py scrape
python cli.py clean
python cli.py train --advanced
python cli.py simulate --iterations 50000
python cli.py analyze
python cli.py predict
```

### Testing

```bash
python -m pytest tests/
python -m pytest tests/test_odds_integration.py
python -m pytest -v tests/
```

### Environment Variables

```bash
export DATABASE_URL="postgresql://user:password@host:port/dbname"  # default: SQLite at data/db/allsvenskan.db
export ODDS_API_KEY="..."  # required for odds integration
```

### Data Pipeline (Python API)

```python
from allsvenskan.data.scraper import AllsvenskanScraper
from allsvenskan.data.cleaner import DataCleaner
from allsvenskan.data.strength import TeamStrengthCalculator
from allsvenskan.models.poisson_model import PoissonModel
from allsvenskan.simulation.simulator import MonteCarloSimulator
from allsvenskan.analysis.aggregator import ResultsAggregator

scraper = AllsvenskanScraper()
raw_data = scraper.scrape_matches(seasons=[2023, 2024])

cleaner = DataCleaner()
results, fixtures = cleaner.clean_data(raw_data)
```

## Architecture

The Python package is `core/` (previously `allsvenskan/`, before that `premier_league/`).

### Data Source

Data comes from **football-data.co.uk** (free, no API key required):
- Primary: `https://www.football-data.co.uk/new/SWE.csv` — aggregated new-format file covering all seasons. Rows with empty HG/AG are upcoming fixtures.
- Fallback: individual season files `https://www.football-data.co.uk/mmz4281/{code}/S1.csv` where code maps from `AllsvenskanScraper.HISTORICAL_SEASONS`.

Column mapping from source: `HG` → `FTHG`, `AG` → `FTAG`. Dates are `dd/mm/yy` format (parsed with `dayfirst=True`). Only rows where `Div == "S1"` (Allsvenskan) are kept.

Allsvenskan runs within a single calendar year (approx. April–November). Season years in the data are e.g. `2024`, not split-year.

### Data Flow

1. **Scraping** (`data/scraper.py`): `AllsvenskanScraper.scrape_matches(seasons=[2024])` → returns DataFrame with Date, HomeTeam, AwayTeam, FTHG, FTAG, Season, SeasonStart
2. **Cleaning** (`data/cleaner.py`): Splits into completed results (FTHG/FTAG filled) and upcoming fixtures (NaN)
3. **Strength** (`data/strength.py`): Computes attack/defense ratings per team
4. **Model** (`models/poisson_model.py`): Poisson regression; `use_mle=True, use_dixon_coles=True` for advanced training
5. **Simulation** (`simulation/simulator.py`): `MonteCarloSimulator(fixtures_df, model).run(n_simulations=10000)`. Use `MonteCarloSimulator.from_upcoming_fixtures(model)` to load from `data/clean/upcoming_fixtures.csv` directly
6. **Analysis** (`analysis/aggregator.py`): `ResultsAggregator` — methods `analyze_results()`, `calculate_championship_odds()`, `calculate_relegation_odds()`, `calculate_expected_points()`
7. **Hybrid** (`models/hybrid_model.py`): Weights Poisson vs. odds dynamically based on season progress (early→70% odds, late→90% Poisson)

### Column Name Standardization

`ColumnStandardizer.standardize_columns(df)` normalizes inconsistent column names across datasets:
- `Home_Team`/`home_team` → `HomeTeam`
- `FTHG`/`home_goals` → `FTHG`

Always run this on loaded DataFrames before processing.

### Streamlit Session State Keys

`data_loaded`, `model_trained`, `simulation_complete`, `db_manager`, `db_connected`, `poisson_model`, `odds_data`, `odds_fetched`, `active_page`

## File Paths

| Path | Contents |
|------|----------|
| `data/clean/results.csv` | Completed matches |
| `data/clean/fixtures.csv` | Upcoming matches (from cleaner) |
| `data/clean/upcoming_fixtures.csv` | Authentic fixture schedule (preferred for simulation) |
| `data/processed/team_stats.csv` | Team strength metrics |
| `models/poisson_params.pkl` | Pickled `PoissonModel` |
| `reports/simulations/sim_results_*.csv` | Monte Carlo output |
| `reports/simulations/fixture_predictions.csv` | Per-match predictions |

## Key Notes

- Odds API (`api.the-odds-api.com`) has request limits on the free tier — only fetch when explicitly requested
- Simulation default is 10,000 iterations; use fewer for dev/testing
- Team names from the football-data.co.uk source and odds API may differ — no automatic mapping exists yet
- Database supports both SQLite (default) and PostgreSQL; schema tables: `matches`, `team_statistics`, `model_parameters`, `simulation_results`, `analysis_results`
- The `premier_league/` directory is a leftover from before the migration and should be deleted once confirmed unnecessary
