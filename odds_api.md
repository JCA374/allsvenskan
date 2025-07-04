import requests
import pandas as pd

# API parameters

ODDS_API_KEY (in secrets)
REGIONS = "eu"  # European bookmakers
MARKETS = "h2h"  # Head-to-head odds
ODDS_FORMAT = "decimal"

# Build the URL
url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
params = {
    "regions": REGIONS,
    "markets": MARKETS,
    "oddsFormat": ODDS_FORMAT,
    "apiKey": API_KEY
}

# Fetch data
response = requests.get(url, params=params)

# Check for success
if response.status_code == 200:
    data = response.json()
    matches = []

    for match in data:
        home = match["home_team"]
        away = match["away_team"]
        commence_time = match["commence_time"]
        bookmakers = match.get("bookmakers", [])

        # Get best available odds (first bookmaker)
        if bookmakers:
            outcomes = bookmakers[0]["markets"][0]["outcomes"]
            odds = {o["name"]: o["price"] for o in outcomes}
        else:
            odds = {}

        matches.append({
            "Home Team": home,
            "Away Team": away,
            "Start Time": commence_time,
            "Home Odds": odds.get(home),
            "Draw Odds": odds.get("Draw"),
            "Away Odds": odds.get(away)
        })

    df = pd.DataFrame(matches)
    display(df)
else:
    print("Error:", response.status_code, response.text)