# Allsvenskan Monte Carlo Forecast

## Overview

This is a comprehensive Allsvenskan (Swedish top-flight football) prediction system that uses Monte Carlo simulation to forecast regular-season outcomes. The application combines data from football-data.co.uk, statistical modeling, and Monte Carlo simulation to project standings, relegation likelihoods, and movement within the table.

The system is built with Python and Streamlit, featuring a modular architecture that separates data collection, cleaning, modeling, simulation, and visualization into distinct components.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Interactive dashboard with multiple pages for different functionalities
- **Plotly Visualizations**: Dynamic charts and graphs for data presentation
- **Session State Management**: Maintains application state across user interactions

### Backend Architecture
- **Modular Python Structure**: Separated concerns across multiple modules
- **Object-Oriented Design**: Each major component implemented as a class
- **Pipeline Architecture**: Data flows through distinct stages: scraping → cleaning → modeling → simulation → analysis

### Data Processing Pipeline
1. **Data Collection**: CSV download from football-data.co.uk (free, no API key required)
2. **Data Cleaning**: Normalization of team names, date parsing, and data validation
3. **Statistical Modeling**: Poisson distribution-based goal prediction model
4. **Monte Carlo Simulation**: Multiple season simulations to generate probability distributions
5. **Results Analysis**: Aggregation and statistical analysis of simulation outcomes

## Key Components

### Data Layer (`allsvenskan/data/`)
- **AllsvenskanScraper**: Downloads Allsvenskan data from football-data.co.uk (`SWE.csv`)
- **DataCleaner**: Normalizes team names, parses dates, splits results from fixtures
- **TeamStrengthCalculator**: Computes attack/defense strengths from historical performance

### Modeling Layer (`allsvenskan/models/`)
- **PoissonModel**: Statistical model for predicting match outcomes using Poisson distribution
- Uses team attack rates, defense rates, and home advantage factors
- Implements Maximum Likelihood Estimation for parameter optimization

### Simulation Layer (`allsvenskan/simulation/`)
- **MonteCarloSimulator**: Runs thousands of season simulations
- Generates probability distributions for final league positions
- Uses randomized goal generation based on Poisson model predictions

### Analysis Layer (`allsvenskan/analysis/`)
- **ResultsAggregator**: Processes simulation results into meaningful statistics
- Calculates championship odds, relegation probabilities, European qualification chances
- Generates position probability matrices and confidence intervals

### Visualization Layer (`allsvenskan/visualization/`)
- **Dashboard**: Streamlit-based interactive dashboard
- Multiple visualization types: bar charts, probability distributions, comparison tables
- Real-time updates based on user interactions

## Data Flow

1. **Scraping**: Match data downloaded from football-data.co.uk (new-format `SWE.csv`)
2. **Cleaning**: Data normalized and split into completed matches (results) and upcoming matches (fixtures)
3. **Storage**: Data persisted in SQLite (default) with optional PostgreSQL support
4. **Strength Calculation**: Team performance metrics calculated from historical results
5. **Model Training**: Poisson model fitted using team strengths and match outcomes
6. **Simulation**: Monte Carlo process simulates remaining fixtures thousands of times
7. **Aggregation**: Results processed to generate statistics and probabilities
8. **Visualization**: Interactive dashboard displays predictions and analysis

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Statistical functions (Poisson distribution, optimization)
- **plotly**: Interactive visualizations

### Web Scraping / Data Fetching
- **requests**: HTTP library for downloading CSV data
- **beautifulsoup4**: HTML parsing (for future use)

### Utilities
- **unidecode**: Unicode normalization for team names
- **pickle**: Model serialization
- **datetime**: Date/time handling

## Deployment Strategy

### Development Environment
- Python-based application designed for local development
- Streamlit development server for rapid iteration
- Modular structure allows for easy testing of individual components

### Data Storage
- **Primary**: Local SQLite database at `data/db/allsvenskan.db` (auto-created on first run)
- **Optional**: Set `DATABASE_URL` to switch to PostgreSQL or any SQLAlchemy-compatible database
- **Fallback**: CSV-based file storage for compatibility and backup
- Database tables: matches, team_statistics, model_parameters, simulation_results, analysis_results
- Directory structure separates raw, cleaned, and processed data
- Results stored in reports directory for persistence

### Scalability Considerations
- Stateless simulation components allow for parallel processing
- Model parameters can be cached and reused
- Results can be pre-computed and cached for faster dashboard loading

## Changelog

```
- Apr 2026. Migrated from SHL (ice hockey) to Allsvenskan (football). New data source: football-data.co.uk.
- Apr 2026. Replaced SHL-specific point rules (OT/SO) with standard football rules (3/1/0).
- Apr 2026. Replaced hardcoded SHL team list in simulator with Allsvenskan team name normalizer.
- Apr 2026. Updated odds config from icehockey_sweden_shl to soccer_sweden_allsvenskan.
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```
