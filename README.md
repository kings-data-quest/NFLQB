# NFL Quarterback Performance Analysis

## Overview
This project implements a data pipeline and analysis focused on quarterback performance metrics from the 2023 NFL regular season. It includes data collection, custom metric development, correlation analysis with team success, and a concept for a real-time data pipeline.

## Project Structure
```
nfl_qb_analysis/
├── data/                      # Data storage directory
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks for analysis
└── src/                       # Source code
    ├── data_collection/       # Scripts to fetch NFL data
    ├── data_processing/       # Data cleaning and preparation
    ├── metric_development/    # Custom QB metric implementation
    ├── analysis/              # Statistical analysis
    └── visualization/         # Data visualization tools
```

## Features
- Data collection from public NFL sources using nfl-data-py (NFLfastR wrapper)
- Comprehensive data cleaning and preprocessing
- Custom QB performance metric incorporating:
  - Passing accuracy at different depths
  - Performance under pressure
  - Mobility and rushing contribution
- Analysis comparing the custom metric to traditional QB ratings
- Correlation analysis with team wins
- Identification of over/underrated QBs
- Concept for a real-time data pipeline

## Installation
1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Data Collection: Run the data collection scripts to fetch 2023 NFL QB data
2. Data Processing: Clean and prepare the data
3. Metric Development: Generate the custom QB performance metric
4. Analysis: Analyze the relationship between the metric and team performance
5. Visualization: Generate visualizations of findings

Alternatively, explore the Jupyter notebooks in the `notebooks/` directory for a guided analysis.

## Data Sources
This project uses data from NFL's public APIs via the nfl-data-py package, a Python wrapper for nflfastR.

## License
[MIT License](LICENSE) 