# Allsvenskan Forecast - CLI Interface Guide

## Quick Start

The CLI provides command-line access to all major features of the Allsvenskan forecasting system.

### Basic Usage

```bash
# Get help
python cli.py --help

# Or make it executable and run directly
./cli.py --help
```

## Available Commands

### 1. Data Scraping
Download Allsvenskan data from football-data.co.uk:

```bash
# Scrape current and previous season (default)
python cli.py scrape

# Scrape specific seasons
python cli.py scrape --seasons 2024 2025

# Scrape multiple seasons
python cli.py scrape --seasons 2022 2023 2024 2025
```

### 2. Data Cleaning
Clean and separate completed matches from upcoming fixtures:

```bash
# Clean most recent scraped data
python cli.py clean

# Clean specific file
python cli.py clean --input data/raw/allsvenskan_matches_20250101_120000.csv
```

### 3. Model Training
Train the Poisson prediction model:

```bash
# Basic training (faster)
python cli.py train

# Advanced training with MLE + Dixon-Coles (slower, more accurate)
python cli.py train --advanced
```

### 4. Monte Carlo Simulation
Run season simulations:

```bash
# Default: 10,000 simulations
python cli.py simulate

# Custom number of simulations
python cli.py simulate --iterations 50000
python cli.py simulate -n 100000

# Quick test (1,000 simulations)
python cli.py simulate -n 1000
```

### 5. Results Analysis
Analyze simulation results and display probabilities:

```bash
# Analyze most recent simulation
python cli.py analyze

# Analyze specific simulation file
python cli.py analyze --input reports/simulations/sim_results_20250101_120000.csv
```

Output includes:
- Championship probabilities for each team
- Relegation probabilities
- Expected final standings by points
- JSON export of full analysis

### 6. Fixture Predictions
Generate predictions for upcoming matches:

```bash
python cli.py predict
```

Shows for each fixture:
- Expected score
- Win/Draw/Loss probabilities
- Saves predictions to CSV

### 7. Full Pipeline
Run the entire workflow in one command:

```bash
# Run everything with defaults
python cli.py pipeline

# Run with custom settings
python cli.py pipeline --advanced --iterations 50000 --seasons 2024 2025
```

Pipeline steps:
1. Scrape data
2. Clean data
3. Train model
4. Run simulation
5. Analyze results
6. Generate predictions

## Command Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--seasons` | - | Seasons to scrape (space-separated) | Current + previous year |
| `--advanced` | - | Use advanced model training | False (basic training) |
| `--iterations` | `-n` | Number of Monte Carlo simulations | 10,000 |
| `--input` | `-i` | Input file path | Most recent file |

## Workflow Examples

### First Time Setup
```bash
# Run the full pipeline
python cli.py pipeline
```

### Daily Update Workflow
```bash
# 1. Scrape latest data
python cli.py scrape

# 2. Clean it
python cli.py clean

# 3. Re-train model with updated data
python cli.py train --advanced

# 4. Run simulation
python cli.py simulate -n 25000

# 5. View analysis
python cli.py analyze

# 6. Check fixture predictions
python cli.py predict
```

### Quick Predictions
If you already have trained model and fixtures:
```bash
python cli.py predict
```

### Development/Testing
Use fewer iterations for faster testing:
```bash
python cli.py simulate -n 100
python cli.py analyze
```

## Output Files

The CLI automatically creates and organizes files:

```
data/
├── raw/              # Scraped data
│   └── allsvenskan_matches_YYYYMMDD_HHMMSS.csv
├── clean/            # Cleaned data
│   ├── results.csv   # Completed matches
│   └── fixtures.csv  # Upcoming matches
└── processed/        # Computed statistics
    └── team_stats.csv

models/
└── poisson_params.pkl  # Trained model

reports/simulations/
├── sim_results_YYYYMMDD_HHMMSS.csv     # Simulation results
├── analysis_YYYYMMDD_HHMMSS.json       # Analysis data
└── fixture_predictions.csv             # Match predictions
```

## Tips

1. **Run pipeline first**: If starting fresh, use `python cli.py pipeline` to set up everything

2. **Simulation speed**:
   - 1,000 iterations: ~10 seconds (testing)
   - 10,000 iterations: ~1-2 minutes (standard)
   - 100,000 iterations: ~10-15 minutes (high accuracy)

3. **Model training**:
   - Basic: Fast, good for development (~5 seconds)
   - Advanced: Slower but more accurate (~30-60 seconds)

4. **Data freshness**: Re-scrape data regularly to get latest match results

5. **Pipeline customization**: The pipeline command accepts all options:
   ```bash
   python cli.py pipeline --advanced -n 50000
   ```

## Troubleshooting

### "No raw data files found"
Run `python cli.py scrape` first

### "Results file not found"
Run `python cli.py clean` first

### "Model file not found"
Run `python cli.py train` first

### "No simulation files found"
Run `python cli.py simulate` first

## Using with Streamlit

The CLI complements the Streamlit web interface. You can:

1. Use CLI for automated workflows and batch processing
2. Use Streamlit UI for interactive exploration and visualization

To run the web interface:
```bash
streamlit run app.py --server.port 5000
```

(Note: Restore app.py from app.py.backup if needed)

## Advanced Usage

### Scripting and Automation

Create bash scripts for automated updates:

```bash
#!/bin/bash
# update_forecasts.sh

echo "Running daily forecast update..."
python cli.py scrape --seasons 2025
python cli.py clean
python cli.py train --advanced
python cli.py simulate -n 25000
python cli.py analyze
python cli.py predict
echo "Update complete!"
```

### Chaining Commands

Use bash operators to chain commands:

```bash
# Run steps in sequence
python cli.py scrape && \
python cli.py clean && \
python cli.py train --advanced && \
python cli.py simulate -n 50000 && \
python cli.py analyze

# Continue even if steps fail (not recommended)
python cli.py scrape ; python cli.py clean ; python cli.py train
```

### Integration with Database

The CLI uses the same database backend as the Streamlit app. Set environment variables:

```bash
# Use PostgreSQL
export DATABASE_URL="postgresql://user:pass@host:port/dbname"
python cli.py pipeline

# Use SQLite (default)
python cli.py pipeline
```

## Getting Help

```bash
# General help
python cli.py --help

# Command-specific examples
python cli.py --help | grep -A 10 "Examples:"
```
