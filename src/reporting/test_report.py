#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for NFL QB Report Generator

This script allows for testing the report generation functionality independently
from the main pipeline.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reporting.report_generator import QBReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NFL QB Report Generator Test')
    
    parser.add_argument('--season', type=int, default=2023,
                        help='NFL season year to analyze (default: 2023)')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for data storage (default: ./data)')
    
    return parser.parse_args()


def main():
    """Run the report generator test."""
    # Parse command line arguments
    args = parse_arguments()
    
    start_time = datetime.now()
    logger.info(f"Starting QB report generation test for {args.season} season")
    
    try:
        # Initialize report generator
        report_generator = QBReportGenerator(data_dir=args.data_dir)
        
        # Generate report
        report_path = report_generator.generate_report(season=args.season)
        
        # Calculate and log execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"Report generation completed successfully in {execution_time}")
        
        print(f"\nNFL QB Analysis Report generated successfully!")
        print(f"Report is available at: {report_path}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        print(f"\nError: Report generation failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 