# Allsvenskan Monte Carlo Forecast

## Overview

This is a comprehensive Swedish football (Allsvenskan) league prediction system that uses Monte Carlo simulation to forecast season outcomes. The application combines web scraping, statistical modeling, and Monte Carlo simulation to predict league table positions, championship odds, relegation probabilities, and European qualification chances.

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
1. **Data Collection**: Web scraping from allsvenskan.se using BeautifulSoup and trafilatura
2. **Data Cleaning**: Normalization of team names, date parsing, and data validation
3. **Statistical Modeling**: Poisson distribution-based goal prediction model
4. **Monte Carlo Simulation**: Multiple season simulations to generate probability distributions
5. **Results Analysis**: Aggregation and statistical analysis of simulation outcomes

## Key Components

### Data Layer (`src/data/`)
- **AllsvenskanScraper**: Web scraper for match data from official website
- **DataCleaner**: Normalizes team names, parses dates, splits results from fixtures
- **TeamStrengthCalculator**: Computes attack/defense strengths from historical performance

### Modeling Layer (`src/models/`)
- **PoissonModel**: Statistical model for predicting match outcomes using Poisson distribution
- Uses team attack rates, defense rates, and home advantage factors
- Implements Maximum Likelihood Estimation for parameter optimization

### Simulation Layer (`src/simulation/`)
- **MonteCarloSimulator**: Runs thousands of season simulations
- Generates probability distributions for final league positions
- Uses randomized goal generation based on Poisson model predictions

### Analysis Layer (`src/analysis/`)
- **ResultsAggregator**: Processes simulation results into meaningful statistics
- Calculates championship odds, relegation probabilities, European qualification chances
- Generates position probability matrices and confidence intervals

### Visualization Layer (`src/visualization/`)
- **Dashboard**: Streamlit-based interactive dashboard
- Multiple visualization types: bar charts, probability distributions, comparison tables
- Real-time updates based on user interactions

## Data Flow

1. **Scraping**: Raw match data extracted from allsvenskan.se
2. **Cleaning**: Data normalized and split into completed matches (results) and upcoming matches (fixtures)
3. **Storage**: Data persisted in PostgreSQL database with file system fallback
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

### Web Scraping
- **requests**: HTTP library for web requests
- **beautifulsoup4**: HTML parsing
- **trafilatura**: Text extraction from web pages

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
- **Primary**: PostgreSQL database for persistent, reliable storage
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
Changelog:
- July 02, 2025. Initial setup
- July 02, 2025. Added PostgreSQL database integration with full CRUD operations
- July 02, 2025. Added Database Management page for monitoring and data viewing
- July 02, 2025. Added Data Verification page for validating scraped and cleaned data
- July 02, 2025. Enhanced scraper to support multi-year data collection (2015-2025) with selectable year ranges
- July 02, 2025. Improved scraper to properly parse Swedish match format (OMGÅNG, dates, venues, scores) and handle 403 errors with better headers and retry logic
- July 02, 2025. Implemented comprehensive fix.md improvements: added detailed logging, fixed silent exception handling, enhanced validation, improved parsing methods, and robust fallback strategies
- July 02, 2025. MAJOR FIX: Identified and addressed web scraper URL issue - allsvenskan.se uses JavaScript to load match data dynamically, making static scraping impossible. Implemented multi-layer approach: API endpoint discovery, Selenium fallback, and comprehensive error handling with clear data source status messages.
- July 02, 2025. COMPLETE REWRITE: Replaced complex web scraping with Football-Data CSV API. Removed 1000+ lines of Selenium/BeautifulSoup code and replaced with simple, reliable API calls. Now downloads 2,500+ authentic matches from trusted source. Updated entire system to use authentic Football-Data team names (AIK, Djurgarden, Malmo FF, etc.) throughout application.
- July 02, 2025. ENHANCED FIXTURE DETECTION: Implemented comprehensive fix.md solution with multi-layer fallback strategy. System now attempts Football-Data API for future fixtures, then Allsvenskan API with enhanced headers, finally generates realistic round-robin fixtures using authentic team names. Monte Carlo simulation now always has meaningful fixtures to work with instead of failing with empty CSV errors.
- July 02, 2025. IMPLEMENTED FIX.MD ENHANCEMENTS: Added current standings integration to Monte Carlo simulation. New features include: calculate_current_standings() and get_current_points_table() helper functions, run_monte_carlo_with_standings() simulator method, simulate_remaining_fixtures_detailed() for fixture predictions, separate_results_and_fixtures() for data separation, enhanced simulation page with current league table display and standings-based simulation option, new "Fixture Predictions" page with detailed match outcome probabilities and expected scores. Simulation now starts from actual current standings rather than zero points for all teams.
- July 02, 2025. ENHANCED LIVE STANDINGS INTEGRATION: Implemented calculate_current_standings_from_url() to fetch real-time Allsvenskan standings directly from Football-Data CSV API. System now displays authentic current league table with live stats (MP, W, D, L, GF, GA, GD, Pts) and uses these actual point totals as starting positions for Monte Carlo simulation. Added current vs projected standings comparison showing how teams are expected to move up/down the table based on remaining fixtures. Enhanced simulation results to show breakdown of current points + simulated points from remaining matches.
- July 03, 2025. UPCOMING FIXTURES INTEGRATION: Integrated user's upcoming_fixtures.csv file directly into Monte Carlo simulation. Implemented robust CSV parsing to handle malformed lines and normalize team names automatically. System now uses authentic upcoming fixtures instead of generated ones when available. Added team name mapping to handle different formats (e.g., "IFK Göteborg" → "Goteborg"). Removed cleaned file creation and works directly with original CSV file. Simulation page shows preview of upcoming fixtures and automatically detects when to use authentic fixture data.
- July 03, 2025. CLEANED UP SYSTEM: Removed incorrect fixtures.csv file per user request. System now exclusively uses authentic upcoming_fixtures.csv file for simulations. Eliminated fallback to generated fixtures to ensure only authentic data is used. Updated simulation logic to require upcoming_fixtures.csv file for operation, maintaining data integrity and authenticity throughout the prediction system.
- July 04, 2025. ODDS INTEGRATION IMPLEMENTATION: Implemented comprehensive odds integration system following odds_implementation_plan.md. Added complete Phase 1 foundation components: OddsRecord schema with Pydantic validation, robust odds converter utilities with margin removal, OddsAPI class for live data fetching from The-Odds-API, HybridPoissonOddsModel combining statistical predictions with betting odds, new "Odds Integration" page in Streamlit app with live odds fetching, hybrid predictions, value bet detection, and model comparison visualizations. System now supports season-progress-based weighting (70% odds weight early season, 10% late season) and identifies value betting opportunities where model probabilities exceed market implications.
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```