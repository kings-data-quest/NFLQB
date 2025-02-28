# NFL Quarterback Performance Analysis

## Overview
This project implements a comprehensive data pipeline and analysis system focused on quarterback performance metrics from the 2023 NFL regular season. It includes data collection, custom metric development, correlation analysis with team success, visualization, and a professional PDF reporting system that presents findings in a format suitable for coaches, analysts, and team management.

## Project Structure
```
nfl_qb_analysis/
├── data/                      # Data storage directory
│   ├── reports/               # Generated PDF reports
│   └── season_2023/           # Season-specific data
│       ├── raw/               # Raw collected data
│       ├── processed/         # Cleaned and processed data
│       ├── metrics/           # Calculated QB metrics
│       ├── analysis/          # Analysis results
│       └── visualizations/    # Generated visualizations
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks for analysis
├── requirements.txt           # Project dependencies
└── src/                       # Source code
    ├── data_collection/       # Scripts to fetch NFL data
    ├── data_processing/       # Data cleaning and preparation
    ├── metric_development/    # Custom QB metric implementation
    ├── analysis/              # Statistical analysis
    ├── visualization/         # Data visualization tools
    ├── reporting/             # PDF report generation
    │   ├── templates/         # Report templates
    │   ├── report_generator.py # Main report generation class
    │   └── test_report.py     # Standalone report testing
    └── main.py                # Main pipeline script
```

## Features
- **Data Collection:** Fetches NFL data including play-by-play, player stats, and team information using nfl-data-py
- **Data Processing:** Cleans, transforms, and enhances raw data for analysis
- **Custom QB Metrics:** Generates a composite QB performance metric incorporating:
  - Passing accuracy at different depths
  - Performance under pressure
  - Mobility and rushing contribution
- **Analysis:** Performs correlation analysis with team success metrics
- **Visualization:** Creates distribution plots, correlation heatmaps, and interactive dashboards
- **Reporting:** Generates comprehensive PDF reports with executive summaries, detailed findings, and actionable recommendations

## Installation
1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Run the Analysis Pipeline
The pipeline is modular and can be run in full or by specific steps.

### Running the Full Pipeline
```bash
python src/main.py --steps all
```

### Running Specific Steps
```bash
# Run only the data collection step
python src/main.py --steps collect

# Run multiple specific steps
python src/main.py --steps collect process metric

# Run just the reporting step (requires previous steps to have been completed)
python src/main.py --steps report
```

### Command Line Arguments
- `--season`: NFL season year to analyze (default: 2023)
- `--data-dir`: Directory for data storage (default: ./data)
- `--steps`: Pipeline steps to run (choices: collect, process, metric, analyze, visualize, report, all)
- `--skip-existing`: Skip steps if output files already exist

## Using the Reporting Module
The reporting module can be run as part of the full pipeline or independently.

### Running the Report Generator Independently
```bash
python src/reporting/test_report.py --season 2023 --data-dir ./data
```

### Report Content
The generated PDF reports include:
1. **Executive Summary:** High-level overview of key findings
2. **Analysis Methodology:** Explanation of data collection and analysis approach
3. **Key Findings:** Detailed insights from the analysis
4. **Detailed QB Analysis:** In-depth performance breakdown
5. **Visual Insights:** Graphical representations of QB metrics
6. **Actionable Recommendations:** Practical suggestions for team management
7. **Real-Time Pipeline Concept:** Proposal for implementing near real-time analysis
8. **Technical Appendix:** Additional details on data sources and methods

## How the Codebase Works

### Pipeline Flow
1. **Data Collection (`src/data_collection/`):**
   - Fetches raw NFL data using nfl-data-py API
   - Saves raw data to `data/season_YYYY/raw/`

2. **Data Processing (`src/data_processing/`):**
   - Cleans and preprocesses raw data
   - Extracts play-by-play information relevant to QB assessment
   - Creates initial metrics like pressure detection
   - Saves processed data to `data/season_YYYY/processed/`

3. **Metric Development (`src/metric_development/`):**
   - Calculates depth-adjusted accuracy metrics
   - Evaluates performance under pressure
   - Assesses mobility contribution
   - Creates composite QB performance score
   - Saves metrics to `data/season_YYYY/metrics/`

4. **Analysis (`src/analysis/`):**
   - Performs correlation analysis between QB metrics and team success
   - Identifies underrated and overrated QBs
   - Generates analysis reports and visualizations
   - Saves analysis to `data/season_YYYY/analysis/`

5. **Visualization (`src/visualization/`):**
   - Creates distribution plots of key metrics
   - Generates correlation heatmaps
   - Builds interactive dashboards
   - Saves visualizations to `data/season_YYYY/visualizations/`

6. **Reporting (`src/reporting/`):**
   - Loads data and analysis results
   - Generates comprehensive PDF reports
   - Integrates text, tables, and visualizations
   - Saves reports to `data/reports/`

### Main Orchestration
The `src/main.py` script orchestrates the entire pipeline, allowing for execution of specific steps or the complete analysis workflow.

### Report Generation
The reporting module (`src/reporting/report_generator.py`) is built on FPDF and provides:
- Custom PDF class with professional header and footer
- Methods for generating each report section
- Integration of data tables and visualizations
- Comprehensive QB performance analysis

## Results and Outputs
After running the pipeline, you can find:
- Raw and processed data in `data/season_YYYY/`
- Generated visualizations in `data/season_YYYY/visualizations/`
- Analysis reports in `data/season_YYYY/analysis/`
- PDF reports in `data/reports/`

## Data Sources
This project uses data from NFL's public APIs via the nfl-data-py package, a Python wrapper for nflfastR.

## License
[MIT License](LICENSE) 