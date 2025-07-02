# collect\_last\_10\_years.md

## Purpose

Provide step-by-step instructions for downloading and filtering Swedish Allsvenskan football results for the last 10 years using the Football-Data CSV feed and pandas.

---

### 1. Install dependencies

Make sure you have pandas installed:

```bash
pip install pandas
```

---

### 2. Download the master CSV

Use the Football-Data CSV URL, which covers Allsvenskan results from 2012 onward:

```python
import pandas as pd

url = "https://www.football-data.co.uk/new/SWE.csv"
df = pd.read_csv(url)
```

---

### 3. Parse dates

Convert the `Date` column to pandas datetime (day-first format):

```python
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
```

---

### 4. Filter for the last 10 years

Compute the cutoff year dynamically and select all matches where the year is greater than or equal to that cutoff:

```python
from datetime import datetime

current_year = datetime.now().year
cutoff = current_year - 10

df_last10 = df[df['Date'].dt.year >= cutoff].reset_index(drop=True)
```

This will include seasons from `cutoff` through the present.

---

### 5. Inspect or save the data

* **View the first few rows**:

  ```python
  df_last10.head()
  ```

* **Save to CSV**:

  ```python
  df_last10.to_csv("allsvenskan_last10_years.csv", index=False)
  ```

---

### 6. (Optional) Further filtering

* **By team** (e.g., Hammarby):

  ```python
  team = "Hammarby"
  mask = (
      df_last10['Home'].str.contains(team, case=False, na=False) |
      df_last10['Away'].str.contains(team, case=False, na=False)
  )
  df_team = df_last10[mask].reset_index(drop=True)
  ```

* **By specific season**:

  ```python
  season = 2023
  df_season = df_last10[df_last10['Date'].dt.year == season]
  ```

---

## Summary

1. Install pandas.
2. Download the full Allsvenskan CSV.
3. Parse dates.
4. Dynamically filter for the last 10 years.
5. Inspect, save, or further filter as needed.




The Football-Data CSV generally ships with the *full season schedule* (even if the scores for future fixtures are blank), so yes—any games still to come should already be in your `df_last10` (or `df_2024`) as long as they’ve been published. You can pull them out by filtering on the date:

```python
import pandas as pd

# (re‐use your df_2024 from before)
today = pd.Timestamp.today().normalize()

# All remaining (future) 2024 fixtures:
upcoming = df_2024[df_2024['Date'] > today].reset_index(drop=True)

# Show them:
upcoming
```

If you see blank score columns (e.g. `FTHG`, `FTAG`, `FTR`) for those rows, that’s expected—they just haven’t been played yet.

---

### If you don’t get future fixtures in the CSV

Some seasons the CSV may only contain completed matches. In that case you can fall back to the league’s JSON schedule endpoint:

```python
import requests

API = "https://allsvenskan.se/wp-json/sef-leagues/v1/matches"
params = {"league": "allsvenskan", "season": 2024}
resp = requests.get(API, params=params)
resp.raise_for_status()
data = resp.json()

# This returns every fixture (past & future) as JSON:
matches = pd.json_normalize(data["matches"])
matches['Date'] = pd.to_datetime(matches['kickoff'], utc=True)
upcoming = matches[matches['Date'] > pd.Timestamp.now(tz='UTC')]
```

Either way, you’ll have a DataFrame of all remaining games.
