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
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```