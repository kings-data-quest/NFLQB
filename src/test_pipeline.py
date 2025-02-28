#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL QB Analysis Pipeline Test Script

This script tests the basic functionality of the NFL QB analysis pipeline.
"""

import os
import sys
import logging
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data_collection.data_fetcher import NFLDataFetcher
from src.data_processing.data_processor import NFLDataProcessor
from src.metric_development.qb_metric import QBMetricDeveloper
from src.analysis.correlation_analysis import QBCorrelationAnalyzer
from src.visualization.visualizer import QBVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_collection(data_dir: str, season: int = 2023) -> bool:
    """
    Test the data collection functionality.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        
    Returns:
        True if test passes, False otherwise
    """
    logger.info("Testing data collection...")
    
    try:
        # Initialize data fetcher
        fetcher = NFLDataFetcher(data_dir=data_dir)
        
        # Test fetching a small amount of data
        # For testing, we'll just fetch player info which is smaller and faster
        player_info = fetcher.get_player_info()
        
        # Check if data was fetched successfully
        if player_info is not None and len(player_info) > 0:
            logger.info(f"Successfully fetched {len(player_info)} player records")
            return True
        else:
            logger.error("Failed to fetch player data")
            return False
    except Exception as e:
        logger.error(f"Error in data collection test: {str(e)}")
        return False


def test_data_processing(data_dir: str, season: int = 2023) -> bool:
    """
    Test the data processing functionality.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        
    Returns:
        True if test passes, False otherwise
    """
    logger.info("Testing data processing...")
    
    try:
        # Initialize data processor
        processor = NFLDataProcessor(data_dir=data_dir)
        
        # Create a small test dataset
        import pandas as pd
        import numpy as np
        
        # Create a minimal test dataset
        player_stats = pd.DataFrame({
            'player_id': ['QB1', 'QB2', 'RB1'],
            'player_name': ['Test QB 1', 'Test QB 2', 'Test RB'],
            'position': ['QB', 'QB', 'RB'],
            'attempts': [200, 150, 0],
            'completions': [120, 90, 0],
            'passing_yards': [1500, 1200, 0],
            'passing_tds': [10, 8, 0],
            'interceptions': [5, 3, 0],
            'recent_team': ['TM1', 'TM2', 'TM1']
        })
        
        pbp_data = pd.DataFrame({
            'play_type': ['pass', 'pass', 'run', 'pass', 'pass'],
            'passer_player_id': ['QB1', 'QB2', np.nan, 'QB1', 'QB2'],
            'complete_pass': [1, 0, np.nan, 1, 1],
            'touchdown': [0, 0, 0, 1, 0],
            'interception': [0, 1, 0, 0, 0],
            'air_yards': [5, 15, np.nan, 25, -2],
            'passing_yards': [10, 0, 0, 30, 2],
            'qb_hit': [0, 1, 0, 0, 0],
            'sack': [0, 0, 0, 0, 0],
            'hurry': [0, 0, 0, 0, 0]
        })
        
        # Test filtering QB data
        qb_stats = processor.filter_qb_data(player_stats)
        if len(qb_stats) != 2:
            logger.error(f"Expected 2 QBs, got {len(qb_stats)}")
            return False
            
        # Test preparing play-by-play data
        prepared_pbp = processor.prepare_pbp_qb_data(pbp_data)
        if 'pass_depth' not in prepared_pbp.columns:
            logger.error("Failed to add pass_depth column")
            return False
            
        logger.info("Data processing test passed")
        return True
    except Exception as e:
        logger.error(f"Error in data processing test: {str(e)}")
        return False


def test_metric_development(data_dir: str, season: int = 2023) -> bool:
    """
    Test the metric development functionality.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        
    Returns:
        True if test passes, False otherwise
    """
    logger.info("Testing metric development...")
    
    try:
        # Initialize metric developer
        metric_developer = QBMetricDeveloper(data_dir=data_dir)
        
        # Create a small test dataset
        import pandas as pd
        
        # Create a minimal test dataset with required columns
        qb_data = pd.DataFrame({
            'player_id': ['QB1', 'QB2'],
            'player_name': ['Test QB 1', 'Test QB 2'],
            'attempts': [200, 150],
            'completions': [120, 90],
            'passing_yards': [1500, 1200],
            'passing_tds': [10, 8],
            'interceptions': [5, 3],
            'behind_los_attempts': [20, 15],
            'behind_los_completions': [18, 14],
            'behind_los_comp_pct': [90, 93.3],
            'short_attempts': [100, 80],
            'short_completions': [70, 55],
            'short_comp_pct': [70, 68.8],
            'medium_attempts': [60, 40],
            'medium_completions': [25, 18],
            'medium_comp_pct': [41.7, 45],
            'deep_attempts': [20, 15],
            'deep_completions': [7, 3],
            'deep_comp_pct': [35, 20],
            'pressure_attempts': [40, 30],
            'pressure_completions': [20, 12],
            'pressure_comp_pct': [50, 40],
            'rushing_yards': [200, 50],
            'rushing_tds': [2, 0],
            'rushing_first_downs': [10, 3]
        })
        
        # Test depth-adjusted accuracy calculation
        enhanced_data = metric_developer.calculate_depth_adjusted_accuracy(qb_data)
        if 'depth_adjusted_comp_pct' not in enhanced_data.columns:
            logger.error("Failed to calculate depth-adjusted accuracy")
            return False
            
        # Test pressure performance calculation
        enhanced_data = metric_developer.calculate_pressure_performance(enhanced_data)
        if 'pressure_performance' not in enhanced_data.columns:
            logger.error("Failed to calculate pressure performance")
            return False
            
        # Test mobility contribution calculation
        enhanced_data = metric_developer.calculate_mobility_contribution(enhanced_data)
        if 'mobility_contribution' not in enhanced_data.columns:
            logger.error("Failed to calculate mobility contribution")
            return False
            
        logger.info("Metric development test passed")
        return True
    except Exception as e:
        logger.error(f"Error in metric development test: {str(e)}")
        return False


def test_correlation_analysis(data_dir: str, season: int = 2023) -> bool:
    """
    Test the correlation analysis functionality.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        
    Returns:
        True if test passes, False otherwise
    """
    logger.info("Testing correlation analysis...")
    
    try:
        # Initialize analyzer
        analyzer = QBCorrelationAnalyzer(data_dir=data_dir)
        
        # Create a small test dataset
        import pandas as pd
        import numpy as np
        
        # Create a minimal test dataset with required columns
        qb_data = pd.DataFrame({
            'player_name': ['QB1', 'QB2', 'QB3', 'QB4', 'QB5'],
            'qb_composite_score': [85, 75, 65, 55, 45],
            'passer_rating': [110, 100, 90, 80, 70],
            'qbr': [75, 65, 55, 45, 35],
            'team_win_pct': [0.8, 0.7, 0.5, 0.4, 0.2],
            'wins': [12, 10, 8, 6, 4],
            'points_for': [450, 400, 350, 300, 250]
        })
        
        # Test correlation calculation
        correlations, p_values = analyzer.calculate_correlations(qb_data)
        
        # Check if correlations were calculated correctly
        if correlations.shape != (3, 3):
            logger.error(f"Expected correlation matrix of shape (3, 3), got {correlations.shape}")
            return False
            
        # Test metric differences analysis
        qb_data['composite_rank'] = qb_data['qb_composite_score'].rank(ascending=False)
        qb_data['passer_rating_rank'] = qb_data['passer_rating'].rank(ascending=False)
        qb_data['qbr_rank'] = qb_data['qbr'].rank(ascending=False)
        qb_data['diff_from_passer_rating'] = qb_data['composite_rank'] - qb_data['passer_rating_rank']
        qb_data['diff_from_qbr'] = qb_data['composite_rank'] - qb_data['qbr_rank']
        
        discrepancies = analyzer.analyze_metric_differences(qb_data)
        
        if 'passer_rating' not in discrepancies:
            logger.error("Failed to analyze metric differences")
            return False
            
        logger.info("Correlation analysis test passed")
        return True
    except Exception as e:
        logger.error(f"Error in correlation analysis test: {str(e)}")
        return False


def test_visualization(data_dir: str, season: int = 2023) -> bool:
    """
    Test the visualization functionality.
    
    Args:
        data_dir: Directory for data storage
        season: NFL season year
        
    Returns:
        True if test passes, False otherwise
    """
    logger.info("Testing visualization...")
    
    try:
        # Initialize visualizer
        visualizer = QBVisualizer(data_dir=data_dir)
        
        # Create a small test dataset
        import pandas as pd
        
        # Create a minimal test dataset with required columns
        qb_data = pd.DataFrame({
            'player_name': ['QB1', 'QB2', 'QB3', 'QB4', 'QB5'],
            'qb_composite_score': [85, 75, 65, 55, 45],
            'passer_rating': [110, 100, 90, 80, 70],
            'qbr': [75, 65, 55, 45, 35],
            'team_win_pct': [0.8, 0.7, 0.5, 0.4, 0.2],
            'wins': [12, 10, 8, 6, 4],
            'depth_adjusted_comp_pct': [70, 65, 60, 55, 50],
            'pressure_performance': [80, 70, 60, 50, 40],
            'mobility_contribution': [90, 70, 50, 30, 10],
            'depth_adjusted_comp_pct_normalized': [100, 75, 50, 25, 0],
            'pressure_performance_normalized': [100, 75, 50, 25, 0],
            'mobility_contribution_normalized': [100, 75, 50, 25, 0],
            'yards_per_attempt_normalized': [100, 75, 50, 25, 0],
            'td_int_ratio_normalized': [100, 75, 50, 25, 0],
            'sack_avoidance_normalized': [100, 75, 50, 25, 0]
        })
        
        # Set up visualization directory
        os.makedirs(os.path.join(data_dir, f'season_{season}', 'visualizations'), exist_ok=True)
        visualizer.viz_dir = os.path.join(data_dir, f'season_{season}', 'visualizations')
        
        # Test top QBs bar chart
        chart_path = visualizer.create_top_qbs_bar_chart(qb_data)
        if not chart_path or not os.path.exists(chart_path):
            logger.error("Failed to create top QBs bar chart")
            return False
            
        # Test metric correlation matrix
        matrix_path = visualizer.create_metric_correlation_matrix(qb_data)
        if not matrix_path or not os.path.exists(matrix_path):
            logger.error("Failed to create metric correlation matrix")
            return False
            
        logger.info("Visualization test passed")
        return True
    except Exception as e:
        logger.error(f"Error in visualization test: {str(e)}")
        return False


def run_tests():
    """Run all tests for the NFL QB analysis pipeline."""
    start_time = datetime.now()
    logger.info("Starting NFL QB analysis pipeline tests")
    
    # Create a temporary test directory
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')
    os.makedirs(test_dir, exist_ok=True)
    
    # Run tests
    tests = {
        'Data Collection': test_data_collection(test_dir),
        'Data Processing': test_data_processing(test_dir),
        'Metric Development': test_metric_development(test_dir),
        'Correlation Analysis': test_correlation_analysis(test_dir),
        'Visualization': test_visualization(test_dir)
    }
    
    # Print test results
    print("\nTest Results:")
    print("=" * 50)
    all_passed = True
    for test_name, result in tests.items():
        status = "PASSED" if result else "FAILED"
        if not result:
            all_passed = False
        print(f"{test_name}: {status}")
    print("=" * 50)
    
    # Print overall result
    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
    
    # Calculate and log execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    logger.info(f"Tests completed in {execution_time}")


if __name__ == "__main__":
    run_tests() 