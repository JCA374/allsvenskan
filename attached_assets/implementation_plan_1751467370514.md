```markdown
# Implementation Plan: Allsvenskan Monte Carlo Forecast

This document outlines the step-by-step implementation plan, data interfaces, and directory structure to get the Allsvenskan mid-season Monte Carlo simulation up and running in Python.

---

## 1. Project Structure

```

allsvenskan-montecarlo/ ├── data/ │   ├── raw/                  # Raw CSV files as downloaded │   │   └── fixtures\_results\_raw\.csv │   ├── clean/                # Cleaned and normalized CSVs │   │   ├── fixtures.csv │   │   └── results.csv │   └── processed/            # Intermediate data for modeling │       └── team\_stats.csv │ ├── notebooks/                # Exploratory notebooks │   └── data\_exploration.ipynb │ ├── src/ │   ├── data/                 # Data ingestion & cleaning │   │   ├── extract.py        # Reads raw CSV │   │   └── clean.py          # Cleaning template & normalization │   │ │   ├── models/               # Model estimation │   │   ├── strength.py       # Attack/defense rate estimators │   │   └── poisson\_model.py  # Poisson/NB model fitting & validation │   │ │   ├── simulation/           # Monte Carlo engine │   │   └── simulate.py       # Season simulator functions │   │ │   ├── analysis/             # Aggregation & diagnostics │   │   └── aggregate.py      # Result summarization tools │   │ │   └── visualization/        # Reporting & charts │       └── plot.py           # Matplotlib routines │ ├── reports/                  # Output tables & plots │   ├── final\_table.csv │   └── charts/ │       ├── position\_heatmap.png │       └── points\_distribution.png │ ├── tests/                    # Unit tests │   ├── test\_clean.py │   └── test\_model.py │ ├── requirements.txt          # Dependencies ├── setup.py                  # Project setup └── README.md

```

---

## 2. Data Interface & Cleaning

### 2.1 Raw Input Format

The raw CSV (`data/raw/fixtures_results_raw.csv`) contains rows like:

```

Date,Venue,Match,HomeGoals,AwayGoals,Summary "LÖRDAG 29 MARS","3arena","Djurgårdens IF - Malmö FF",0,1,"Summering" ... (subsequent rounds)

````

### 2.2 Cleaning Template (`src/data/clean.py`)

```python
import pandas as pd
import unidecode
from datetime import datetime

# 1. Load raw file
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# 2. Normalize columns and types
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase and remove accents from team names
    df['Match'] = df['Match'].apply(lambda x: unidecode.unidecode(x))
    # Split home/away
    teams = df['Match'].str.split(' - ', expand=True)
    df['HomeTeam'] = teams[0]
    df['AwayTeam'] = teams[1]

    # Parse date (Swedish month names)
    month_map = {
        'JANUARI':'January','FEBRUARI':'February','MARS':'March',
        'APRIL':'April','MAJ':'May','JUNI':'June',
        'JULI':'July','AUGUSTI':'August',/*...*/
    }
    df['Date'] = df['Date'].str.upper().replace(month_map, regex=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%A %d %B')
    
    # Rename goals
    df.rename(columns={'HomeGoals': 'FTHG', 'AwayGoals': 'FTAG'}, inplace=True)

    # Filter past vs. future
    results = df.dropna(subset=['FTHG','FTAG'])
    fixtures = df[df['FTHG'].isna() | df['FTAG'].isna()].copy()

    return results[['Date','HomeTeam','AwayTeam','FTHG','FTAG']], \
           fixtures[['Date','HomeTeam','AwayTeam']]

# 3. Save cleaned outputs
def save_clean(results: pd.DataFrame, fixtures: pd.DataFrame, out_dir: str):
    results.to_csv(f"{out_dir}/results.csv", index=False)
    fixtures.to_csv(f"{out_dir}/fixtures.csv", index=False)

if __name__ == '__main__':
    raw = load_raw('data/raw/fixtures_results_raw.csv')
    results, fixtures = clean_raw(raw)
    save_clean(results, fixtures, 'data/clean')
````

**Output Interfaces**:

- `data/clean/results.csv` → consumed by `src/data/strength.py` and `poisson_model.py`
- `data/clean/fixtures.csv` → input to `src/simulation/simulate.py`

---

## 3. Model Estimation

### 3.1 Team Stats Generator (`src/data/strength.py`)

- **Input**: `data/clean/results.csv`
- **Output**: `data/processed/team_stats.csv`

```python
import pandas as pd

def compute_rates(results: pd.DataFrame) -> pd.DataFrame:
    # Group by team, home/away
    home = results.groupby('HomeTeam').agg(
        home_scored=('FTHG','mean'),
        home_conceded=('FTAG','mean'),
        home_games=('FTHG','count')
    )
    away = results.groupby('AwayTeam').agg(
        away_scored=('FTAG','mean'),
        away_conceded=('FTHG','mean'),
        away_games=('FTAG','count')
    )
    stats = home.join(away, how='outer').fillna(0)
    stats['league_avg'] = (results['FTHG'].sum() + results['FTAG'].sum()) \
        / (len(results) * 2)
    return stats

if __name__ == '__main__':
    df = pd.read_csv('data/clean/results.csv')
    stats = compute_rates(df)
    stats.to_csv('data/processed/team_stats.csv')
```

### 3.2 Poisson Model Fitter (`src/models/poisson_model.py`)

- **Input**: `data/processed/team_stats.csv`
- **Output**: Pickled model parameters (`models/poisson_params.pkl`)

```python
import pandas as pd
import pickle

def fit_poisson(stats: pd.DataFrame):
    # Compute mu_home and mu_away per fixture
    # (Load fixtures and merge stats)
    # Use MLE or direct formulas to calibrate home_advantage
    params = { 'home_adv': 1.1, 'attack_rates': {...}, 'defense_rates': {...} }
    return params

if __name__ == '__main__':
    stats = pd.read_csv('data/processed/team_stats.csv')
    params = fit_poisson(stats)
    pickle.dump(params, open('models/poisson_params.pkl','wb'))
```

---

## 4. Simulation Engine (`src/simulation/simulate.py`)

- **Input**: `data/clean/fixtures.csv`, `models/poisson_params.pkl`
- **Output**: Simulation results (in-memory or saved to `reports/simulations/`)

```python
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson

class MonteCarloSimulator:
    def __init__(self, fixtures, params, seed=42):
        self.fixtures = fixtures
        self.params = params
        self.rng = np.random.RandomState(seed)

    def simulate_one(self):
        table = {team: 0 for team in self.params['attack_rates']}
        # simulate each match
        for _, row in self.fixtures.iterrows():
            mu_h = ...  # compute from params
            mu_a = ...
            gh = poisson.rvs(mu_h, random_state=self.rng)
            ga = poisson.rvs(mu_a, random_state=self.rng)
            if gh>ga: table[row.HomeTeam] += 3
            elif gh<ga: table[row.AwayTeam] += 3
            else: table[row.HomeTeam] += 1; table[row.AwayTeam] += 1
        return table

    def run(self, n=10000):
        sims = []
        for i in range(n):
            sims.append(self.simulate_one())
        return pd.DataFrame(sims)

if __name__ == '__main__':
    fixtures = pd.read_csv('data/clean/fixtures.csv')
    params = pickle.load(open('models/poisson_params.pkl','rb'))
    sim = MonteCarloSimulator(fixtures, params)
    results = sim.run(10000)
    results.to_csv('reports/simulations/sim_results.csv', index=False)
```

---

## 5. Aggregation & Visualization

- **Aggregation** (`src/analysis/aggregate.py`): read `sim_results.csv`, compute position probabilities, expected points, save `reports/final_table.csv`.
- **Visuals** (`src/visualization/plot.py`): load aggregated output and create charts in `reports/charts/`.

---

## 6. Testing & CI

- **Unit tests** in `tests/` for each module:

  - Data cleaning produces expected number of rows and columns.
  - Model outputs non-negative rates.
  - Simulator conserves total points per season.

- **CI pipeline**: GitHub Actions that runs `pytest` and lints with `flake8` on each push.

---

### Next Steps

1. Scaffold repository & virtual environment.
2. Wire up data cleaning and test on one raw CSV.
3. Implement model estimation and validate on sample data.
4. Build simulator and run a few test simulations.
5. Aggregate results, finalize reports, and review outputs.
6. Add optional enhancements (bookmaker calibration, xG).

*Ready to code!*

```
```
