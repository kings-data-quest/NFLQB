#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL QB Analysis Pipeline Main Script

This script orchestrates the entire NFL QB analysis pipeline, from data collection
to visualization and reporting.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nfl_qb_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data_collection.data_fetcher import NFLDataFetcher
from data_processing.data_processor import NFLDataProcessor
from metric_development.qb_metric import QBMetricDeveloper
from analysis.correlation_analysis import QBCorrelationAnalyzer
from visualization.visualizer import QBVisualizer
from reporting.report_generator import QBReportGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NFL QB Analysis Pipeline')
    
    parser.add_argument('--season', type=int, default=2023,
                        help='NFL season year to analyze (default: 2023)')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for data storage (default: ./data)')
    
    parser.add_argument('--steps', type=str, nargs='+',
                        choices=['collect', 'process', 'metric', 'analyze', 'visualize', 'report', 'all'],
                        default=['all'],
                        help='Pipeline steps to run (default: all)')
    
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip steps if output files already exist')
    
    return parser.parse_args()


def run_data_collection(data_dir: str, season: int, skip_existing: bool = False) -> Dict:
    """
    Run the data collection step of the pipeline.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        skip_existing: Skip if output files already exist
        
    Returns:
        Dictionary of collected datasets
    """
    logger.info("Starting data collection step")
    
    # Check if output files already exist
    season_dir = os.path.join(data_dir, f'season_{season}', 'raw')
    if skip_existing and os.path.exists(season_dir) and len(os.listdir(season_dir)) > 0:
        logger.info(f"Raw data files already exist in {season_dir}, skipping collection")
        return {}
    
    # Initialize data fetcher
    fetcher = NFLDataFetcher(data_dir=data_dir)
    
    # Fetch data
    try:
        raw_data = fetcher.get_qb_data(season=season)
        logger.info("Data collection completed successfully")
        return raw_data
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        raise


def run_data_processing(data_dir: str, season: int, raw_data: Optional[Dict] = None, 
                       skip_existing: bool = False) -> Dict:
    """
    Run the data processing step of the pipeline.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        raw_data: Optional dictionary of raw datasets
        skip_existing: Skip if output files already exist
        
    Returns:
        Processed QB data
    """
    logger.info("Starting data processing step")
    
    # Check if output files already exist
    processed_dir = os.path.join(data_dir, f'season_{season}', 'processed')
    output_file = os.path.join(processed_dir, 'qb_analysis_data.csv')
    
    if skip_existing and os.path.exists(output_file):
        logger.info(f"Processed data file already exists at {output_file}, skipping processing")
        processor = NFLDataProcessor(data_dir=data_dir)
        return processor.load_raw_data(season=season)
    
    # Initialize data processor
    processor = NFLDataProcessor(data_dir=data_dir)
    
    try:
        # Load raw data if not provided
        if raw_data is None or not raw_data:
            raw_data = processor.load_raw_data(season=season)
        
        # Process data
        processed_data = processor.process_qb_data(raw_data)
        logger.info("Data processing completed successfully")
        return processed_data
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise


def run_metric_development(data_dir: str, season: int, processed_data: Optional[Dict] = None,
                          skip_existing: bool = False) -> Dict:
    """
    Run the metric development step of the pipeline.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        processed_data: Optional processed QB data
        skip_existing: Skip if output files already exist
        
    Returns:
        QB data with metrics
    """
    logger.info("Starting metric development step")
    
    # Check if output files already exist
    metrics_dir = os.path.join(data_dir, f'season_{season}', 'metrics')
    output_file = os.path.join(metrics_dir, 'qb_composite_metrics.csv')
    
    if skip_existing and os.path.exists(output_file):
        logger.info(f"Metrics data file already exists at {output_file}, skipping metric development")
        metric_developer = QBMetricDeveloper(data_dir=data_dir)
        return metric_developer.load_processed_data(season=season)
    
    # Initialize metric developer
    metric_developer = QBMetricDeveloper(data_dir=data_dir)
    
    try:
        # Develop QB metrics
        qb_metrics = metric_developer.develop_qb_metric(processed_data, season=season)
        logger.info("Metric development completed successfully")
        return qb_metrics
    except Exception as e:
        logger.error(f"Error in metric development: {str(e)}")
        raise


def run_correlation_analysis(data_dir: str, season: int, qb_metrics: Optional[Dict] = None,
                            skip_existing: bool = False) -> Dict:
    """
    Run the correlation analysis step of the pipeline.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        qb_metrics: Optional QB metrics data
        skip_existing: Skip if output files already exist
        
    Returns:
        Analysis results
    """
    logger.info("Starting correlation analysis step")
    
    # Check if output files already exist
    analysis_dir = os.path.join(data_dir, f'season_{season}', 'analysis')
    output_file = os.path.join(analysis_dir, 'correlation_analysis_report.md')
    
    if skip_existing and os.path.exists(output_file):
        logger.info(f"Analysis report already exists at {output_file}, skipping analysis")
        return {}
    
    # Initialize analyzer
    analyzer = QBCorrelationAnalyzer(data_dir=data_dir)
    
    try:
        # Run correlation analysis
        analysis_results = analyzer.run_correlation_analysis(qb_metrics, season=season)
        logger.info("Correlation analysis completed successfully")
        return analysis_results
    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        raise


def run_visualization(data_dir: str, season: int, qb_metrics: Optional[Dict] = None,
                     skip_existing: bool = False) -> Dict:
    """
    Run the visualization step of the pipeline.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        qb_metrics: Optional QB metrics data
        skip_existing: Skip if output files already exist
        
    Returns:
        Visualization paths
    """
    logger.info("Starting visualization step")
    
    # Check if output files already exist
    viz_dir = os.path.join(data_dir, f'season_{season}', 'visualizations')
    
    if skip_existing and os.path.exists(viz_dir) and len(os.listdir(viz_dir)) > 0:
        logger.info(f"Visualization files already exist in {viz_dir}, skipping visualization")
        return {}
    
    # Initialize visualizer
    visualizer = QBVisualizer(data_dir=data_dir)
    
    try:
        # Create visualizations
        viz_paths = visualizer.create_visualizations(qb_metrics, season=season)
        logger.info("Visualization completed successfully")
        return viz_paths
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        raise


def run_reporting(data_dir: str, season: int, skip_existing: bool = False) -> str:
    """
    Run the reporting step of the pipeline.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        skip_existing: Skip if output files already exist
        
    Returns:
        Path to the generated report
    """
    logger.info("Starting reporting step")
    
    # Check if output files already exist
    reports_dir = os.path.join(data_dir, 'reports')
    
    if skip_existing and os.path.exists(reports_dir) and len(os.listdir(reports_dir)) > 0:
        report_files = [f for f in os.listdir(reports_dir) if f.startswith(f'nfl_qb_analysis_report_{season}')]
        if report_files:
            logger.info(f"Report files already exist in {reports_dir}, skipping reporting")
            return os.path.join(reports_dir, report_files[0])
    
    # Initialize report generator
    report_generator = QBReportGenerator(data_dir=data_dir)
    
    try:
        # Generate report
        report_path = report_generator.generate_report(season=season)
        logger.info(f"Reporting completed successfully, report saved to {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error in reporting: {str(e)}")
        raise


def run_pipeline(args):
    """
    Run the complete NFL QB analysis pipeline based on provided arguments.
    
    Args:
        args: Command line arguments
    """
    start_time = datetime.now()
    logger.info(f"Starting NFL QB analysis pipeline for {args.season} season")
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Determine which steps to run
    run_all = 'all' in args.steps
    steps_to_run = {
        'collect': run_all or 'collect' in args.steps,
        'process': run_all or 'process' in args.steps,
        'metric': run_all or 'metric' in args.steps,
        'analyze': run_all or 'analyze' in args.steps,
        'visualize': run_all or 'visualize' in args.steps,
        'report': run_all or 'report' in args.steps
    }
    
    # Run pipeline steps
    raw_data = None
    processed_data = None
    qb_metrics = None
    report_path = None
    
    try:
        # Step 1: Data Collection
        if steps_to_run['collect']:
            raw_data = run_data_collection(args.data_dir, args.season, args.skip_existing)
        
        # Step 2: Data Processing
        if steps_to_run['process']:
            processed_data = run_data_processing(args.data_dir, args.season, raw_data, args.skip_existing)
        
        # Step 3: Metric Development
        if steps_to_run['metric']:
            qb_metrics = run_metric_development(args.data_dir, args.season, processed_data, args.skip_existing)
        
        # Step 4: Correlation Analysis
        if steps_to_run['analyze']:
            run_correlation_analysis(args.data_dir, args.season, qb_metrics, args.skip_existing)
        
        # Step 5: Visualization
        if steps_to_run['visualize']:
            run_visualization(args.data_dir, args.season, qb_metrics, args.skip_existing)
        
        # Step 6: Reporting
        if steps_to_run['report']:
            report_path = run_reporting(args.data_dir, args.season, args.skip_existing)
        
        # Calculate and log execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"Pipeline completed successfully in {execution_time}")
        
        print("\nNFL QB Analysis Pipeline completed successfully!")
        print(f"Results are available in: {args.data_dir}/season_{args.season}/")
        if report_path:
            print(f"PDF Report is available at: {report_path}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\nError: Pipeline execution failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the pipeline
    run_pipeline(args)