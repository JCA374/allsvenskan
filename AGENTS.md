# Repository Guidelines

## Project Structure & Module Organization
- `app.py` hosts the Streamlit UI; each navigation tab calls a page-level function.  
- Allsvenskan-specific logic is organized under `core/`: scraping and cleaning in `core/data/`, modeling in `core/models/`, simulation engines in `core/simulation/`, analytics in `core/analysis/`, visualization helpers in , and database utilities in `core/database/`.  
- Tests mirror this layout inside `tests/`. Reusable notebooks or scripts live in `reports/` and `scripts/`.  
- Data artifacts are kept in `data/` (`raw/`, `clean/`, `db/`), with the default SQLite file stored at `data/db/allsvenskan.db`.

## Build, Test, and Development Commands
- `uv pip install -r requirements.txt` (or `pip install -r requirements.txt`) — install Python 3.11+ dependencies.  
- `streamlit run app.py` — launch the local dashboard at `http://localhost:8501`.  
- `pytest` — execute the automated suite; set `PYTHONPATH=.` if your shell needs explicit path configuration.  
- `python -m scripts.backfill_results` — example pattern for running maintenance utilities from repo root.

## Coding Style & Naming Conventions
- Follow PEP 8: four-space indentation, `snake_case` functions, `PascalCase` classes, constants in `UPPER_CASE`.  
- Prefer explicit type hints and docstrings on public functions, especially when exposing data structures to Streamlit or simulation modules.  
- Keep filenames descriptive (`allsvenskan_scraper.py`, `team_strength.py`); avoid non-standard abbreviations.

## Testing Guidelines
- Tests use `pytest`; place module-specific coverage in parallel directories (e.g., `tests/data/test_scraper.py`).  
- Start each test with `test_` and separate Arrange/Act/Assert sections with blank lines for readability.  
- Provide fixtures or sample CSVs in `tests/fixtures/` when validating new scrapers, simulations, or odds integrations.  
- Target scenarios that cover missing fixtures, API fallbacks, and database writes.

## Commit & Pull Request Guidelines
- Commit messages should be short and imperative (`Add Allsvenskan standings helper`, `Fix relegation zone calculation`).  
- Scope each commit to a single concern; split refactors from feature additions.  
- Pull requests must outline intent, key changes, and verification steps (e.g., `pytest`, manual Streamlit checks); include screenshots or GIFs for UI updates and note any new environment variables.

## Security & Configuration Tips
- Store credentials via environment variables (`DATABASE_URL`, `ODDS_API_KEY`); never commit secrets.  
- Reset or migrate `data/db/allsvenskan.db` when schema changes occur, and document the process in the PR.  
- Data is scraped from `football-data.co.uk`; keep polite headers. Monitor The Odds API usage limits when enabling live odds fetching.
