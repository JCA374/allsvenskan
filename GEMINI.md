# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the Allsvenskan prediction project.

## Project Overview

This project is a Python-based application for forecasting Allsvenskan (Swedish top-flight football) outcomes. It uses a Streamlit web interface to control the process of data scraping, cleaning, model training, simulation, and visualization.

The core of the project is a Poisson model that predicts the outcome of football games. The model is trained on historical data scraped from the official football-data.co.uk. The application can also integrate betting odds to create a more accurate hybrid model.

### Key Technologies

*   **Python:** The core programming language.
*   **Streamlit:** Used to create the interactive web application.
*   **Pandas:** For data manipulation and analysis.
*   **scikit-learn, scipy, numpy:** For statistical modeling and numerical operations.
*   **BeautifulSoup, requests:** For web scraping.
*   **Plotly:** For data visualization.
*   **SQLAlchemy:** For database interaction (PostgreSQL).

### Architecture

The project is structured into several components:

1.  **Data Scraping (`allsvenskan.data.scraper`):** Downloads match data from football-data.co.uk.
2.  **Data Cleaning (`allsvenskan.data.cleaner`):** Cleans and preprocesses the scraped data.
3.  **Team Strength Calculation (`allsvenskan.data.strength`):** Calculates team attack and defense strengths based on historical performance.
4.  **Poisson Model (`allsvenskan.models.poisson_model`):** A Poisson distribution-based model to predict match outcomes. It can be trained with or without Maximum Likelihood Estimation (MLE).
5.  **Hybrid Model (`allsvenskan.models.hybrid_model`):** Combines the Poisson model with betting odds for improved predictions.
6.  **Monte Carlo Simulation (`allsvenskan.simulation.simulator`):** Runs thousands of simulations of the remaining season to forecast final league standings.
7.  **Streamlit App (`app.py`):** Provides a user interface to control the workflow and visualize the results.
8.  **Database (`allsvenskan.database.db_manager`):** Manages the storage of data in a PostgreSQL database.

## Building and Running

### 1. Setup Environment

To set up the development environment, install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Running the Application

The main application is a Streamlit app. To run it, use the following command:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your browser.

## Development Conventions

### Code Structure

The main application logic is contained within the `allsvenskan` directory, which is structured as a Python package. This promotes modularity and code reuse.

*   `allsvenskan/data`: Modules for data scraping, cleaning, and processing.
*   `allsvenskan/models`: Contains the Poisson and hybrid prediction models.
*   `allsvenskan/simulation`: The Monte Carlo simulator.
*   `allsvenskan/visualization`: Modules for creating the dashboard.
*   `allsvenskan/database`: Database management code.
*   `allsvenskan/utils`: Helper functions and utilities.

### Testing

The project includes a `tests` directory, indicating that unit tests are part of the development process. Tests are written using `pytest`. To run the tests, you can use the following command:

```bash
pytest
```
