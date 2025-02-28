#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL QB Metric Development Module

This module implements a custom quarterback performance metric that incorporates
passing accuracy at different depths, performance under pressure, and mobility/rushing contribution.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QBMetricDeveloper:
    """Class for developing and calculating custom QB performance metrics."""

    def __init__(self, data_dir: str = '../data'):
        """
        Initialize the QB metric developer.

        Args:
            data_dir: Directory containing processed data and where metric results will be stored
        """
        self.data_dir = data_dir
        logger.info(f"Initialized QB metric developer with data directory: {data_dir}")

    def load_processed_data(self, season: int = 2023) -> pd.DataFrame:
        """
        Load processed QB data for the specified season.

        Args:
            season: NFL season year

        Returns:
            DataFrame containing processed QB data
        """
        processed_dir = os.path.join(self.data_dir, f'season_{season}', 'processed')
        file_path = os.path.join(processed_dir, 'qb_analysis_data.csv')
        
        try:
            qb_data = pd.read_csv(file_path)
            logger.info(f"Loaded processed QB data with {len(qb_data)} records")
            return qb_data
        except Exception as e:
            logger.error(f"Error loading processed QB data: {str(e)}")
            raise

    def calculate_depth_adjusted_accuracy(self, qb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate depth-adjusted passing accuracy.
        
        This metric weights completion percentage by pass depth, giving more credit
        for completing passes at greater depths.

        Args:
            qb_data: DataFrame with QB statistics

        Returns:
            DataFrame with added depth-adjusted accuracy metric
        """
        # Create a copy to avoid modifying the original
        enhanced_data = qb_data.copy()
        
        # Define weights for different pass depths
        depth_weights = {
            'behind_los': 0.5,   # Behind line of scrimmage passes are easiest
            'short': 1.0,        # Short passes (0-9 yards)
            'medium': 1.5,       # Medium passes (10-19 yards)
            'deep': 2.0          # Deep passes (20+ yards) are most difficult
        }
        
        # Calculate weighted completion percentage for each QB
        for qb_idx, qb in enhanced_data.iterrows():
            total_weighted_completions = 0
            total_weighted_attempts = 0
            
            for depth, weight in depth_weights.items():
                attempts = qb[f'{depth}_attempts']
                comp_pct = qb[f'{depth}_comp_pct']
                
                if attempts > 0:
                    completions = attempts * (comp_pct / 100)
                    
                    # Weight both attempts and completions
                    weighted_attempts = attempts * weight
                    weighted_completions = completions * weight
                    
                    total_weighted_completions += weighted_completions
                    total_weighted_attempts += weighted_attempts
            
            # Calculate the depth-adjusted completion percentage
            if total_weighted_attempts > 0:
                depth_adjusted_comp_pct = (total_weighted_completions / total_weighted_attempts) * 100
            else:
                depth_adjusted_comp_pct = 0
                
            enhanced_data.loc[qb_idx, 'depth_adjusted_comp_pct'] = depth_adjusted_comp_pct
        
        logger.info("Calculated depth-adjusted accuracy for all QBs")
        return enhanced_data

    def calculate_pressure_performance(self, qb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a metric for QB performance under pressure.
        
        This metric evaluates how well a QB performs when under pressure compared
        to their normal performance.

        Args:
            qb_data: DataFrame with QB statistics

        Returns:
            DataFrame with added pressure performance metric
        """
        # Create a copy to avoid modifying the original
        enhanced_data = qb_data.copy()
        
        # Calculate pressure performance metric
        for qb_idx, qb in enhanced_data.iterrows():
            # Regular completion percentage
            regular_comp_pct = qb['completions'] / qb['attempts'] * 100 if qb['attempts'] > 0 else 0
            
            # Completion percentage under pressure
            pressure_comp_pct = qb['pressure_comp_pct']
            
            # Calculate the ratio (how QB performs under pressure vs. normal)
            # A value of 1.0 means they perform the same under pressure
            # Values > 1.0 mean they perform better under pressure (rare)
            # Values < 1.0 mean they perform worse under pressure (common)
            if regular_comp_pct > 0:
                pressure_ratio = pressure_comp_pct / regular_comp_pct
            else:
                pressure_ratio = 0
                
            # Normalize to a 0-100 scale for easier interpretation
            # A perfect score of 100 would mean they perform the same or better under pressure
            # Most QBs will score lower since performance typically declines under pressure
            pressure_score = min(pressure_ratio * 100, 100)
            
            enhanced_data.loc[qb_idx, 'pressure_performance'] = pressure_score
        
        logger.info("Calculated pressure performance metric for all QBs")
        return enhanced_data

    def calculate_mobility_contribution(self, qb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a metric for QB mobility and rushing contribution.
        
        This metric evaluates how much a QB contributes to the offense with their legs,
        including rushing yards, touchdowns, and first downs.

        Args:
            qb_data: DataFrame with QB statistics

        Returns:
            DataFrame with added mobility contribution metric
        """
        # Create a copy to avoid modifying the original
        enhanced_data = qb_data.copy()
        
        # Ensure required columns exist
        required_cols = ['rushing_yards', 'rushing_tds', 'rushing_first_downs']
        for col in required_cols:
            if col not in enhanced_data.columns:
                enhanced_data[col] = 0
        
        # Calculate mobility score
        for qb_idx, qb in enhanced_data.iterrows():
            # Components of mobility score
            rush_yards = qb['rushing_yards']
            rush_tds = qb['rushing_tds'] * 20  # Weight TDs more heavily
            rush_first_downs = qb['rushing_first_downs'] * 5  # Weight first downs
            
            # Raw mobility score
            mobility_raw = rush_yards + rush_tds + rush_first_downs
            
            enhanced_data.loc[qb_idx, 'mobility_raw'] = mobility_raw
        
        # Normalize mobility score to 0-100 scale
        if len(enhanced_data) > 0:
            max_mobility = enhanced_data['mobility_raw'].max()
            if max_mobility > 0:
                enhanced_data['mobility_contribution'] = (enhanced_data['mobility_raw'] / max_mobility) * 100
            else:
                enhanced_data['mobility_contribution'] = 0
        
        logger.info("Calculated mobility contribution metric for all QBs")
        return enhanced_data

    def calculate_efficiency_metrics(self, qb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional efficiency metrics for QBs.
        
        This includes metrics like yards per attempt, TD-INT ratio, and sack avoidance.

        Args:
            qb_data: DataFrame with QB statistics

        Returns:
            DataFrame with added efficiency metrics
        """
        # Create a copy to avoid modifying the original
        enhanced_data = qb_data.copy()
        
        # Calculate yards per attempt
        enhanced_data['yards_per_attempt'] = enhanced_data['passing_yards'] / enhanced_data['attempts']
        enhanced_data['yards_per_attempt'] = enhanced_data['yards_per_attempt'].fillna(0)
        
        # Calculate TD-INT ratio (with adjustment to avoid division by zero)
        enhanced_data['td_int_ratio'] = enhanced_data['passing_tds'] / (enhanced_data['interceptions'] + 0.5)
        enhanced_data['td_int_ratio'] = enhanced_data['td_int_ratio'].fillna(0)
        
        # Calculate sack avoidance (lower sack percentage is better)
        if 'sacks' in enhanced_data.columns and 'dropbacks' in enhanced_data.columns:
            enhanced_data['sack_pct'] = (enhanced_data['sacks'] / enhanced_data['dropbacks']) * 100
            enhanced_data['sack_pct'] = enhanced_data['sack_pct'].fillna(0)
            
            # Invert sack percentage so higher is better (for consistent scoring)
            max_sack_pct = enhanced_data['sack_pct'].max()
            if max_sack_pct > 0:
                enhanced_data['sack_avoidance'] = 100 - (enhanced_data['sack_pct'] / max_sack_pct * 100)
            else:
                enhanced_data['sack_avoidance'] = 100
        else:
            enhanced_data['sack_avoidance'] = 50  # Default middle value if data not available
        
        logger.info("Calculated efficiency metrics for all QBs")
        return enhanced_data

    def normalize_metric_components(self, qb_data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Normalize metric components to a 0-100 scale.

        Args:
            qb_data: DataFrame with QB statistics
            columns: List of columns to normalize

        Returns:
            DataFrame with normalized metric components
        """
        # Create a copy to avoid modifying the original
        normalized_data = qb_data.copy()
        
        # Initialize scaler
        scaler = MinMaxScaler(feature_range=(0, 100))
        
        # Normalize each column
        for col in columns:
            if col in normalized_data.columns:
                # Handle columns with all zeros
                if normalized_data[col].max() == normalized_data[col].min():
                    normalized_data[f'{col}_normalized'] = 0
                else:
                    # Reshape for scaler
                    values = normalized_data[col].values.reshape(-1, 1)
                    normalized_values = scaler.fit_transform(values)
                    normalized_data[f'{col}_normalized'] = normalized_values
        
        logger.info(f"Normalized {len(columns)} metric components")
        return normalized_data

    def calculate_composite_qb_metric(self, qb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the composite QB performance metric.
        
        This combines all the individual metrics into a single comprehensive score.

        Args:
            qb_data: DataFrame with QB statistics

        Returns:
            DataFrame with added composite QB metric
        """
        # Step 1: Calculate individual component metrics
        enhanced_data = self.calculate_depth_adjusted_accuracy(qb_data)
        enhanced_data = self.calculate_pressure_performance(enhanced_data)
        enhanced_data = self.calculate_mobility_contribution(enhanced_data)
        enhanced_data = self.calculate_efficiency_metrics(enhanced_data)
        
        # Step 2: Normalize all component metrics to 0-100 scale
        components = [
            'depth_adjusted_comp_pct',
            'pressure_performance',
            'mobility_contribution',
            'yards_per_attempt',
            'td_int_ratio',
            'sack_avoidance'
        ]
        normalized_data = self.normalize_metric_components(enhanced_data, components)
        
        # Step 3: Define weights for each component in the composite metric
        weights = {
            'depth_adjusted_comp_pct_normalized': 0.30,  # Passing accuracy is most important
            'pressure_performance_normalized': 0.20,     # Performance under pressure
            'mobility_contribution_normalized': 0.15,    # Mobility/rushing contribution
            'yards_per_attempt_normalized': 0.15,        # Efficiency
            'td_int_ratio_normalized': 0.15,            # Decision making
            'sack_avoidance_normalized': 0.05           # Pocket awareness
        }
        
        # Step 4: Calculate weighted sum for composite metric
        normalized_data['qb_composite_score'] = 0
        
        for component, weight in weights.items():
            if component in normalized_data.columns:
                normalized_data['qb_composite_score'] += normalized_data[component] * weight
        
        # Step 5: Round to 2 decimal places for readability
        normalized_data['qb_composite_score'] = normalized_data['qb_composite_score'].round(2)
        
        logger.info("Calculated composite QB performance metric for all QBs")
        return normalized_data

    def compare_with_traditional_metrics(self, qb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compare the custom QB metric with traditional QB rating systems.
        
        This adds columns showing the difference between rankings in different systems.

        Args:
            qb_data: DataFrame with QB statistics including the composite score

        Returns:
            DataFrame with added comparison metrics
        """
        # Create a copy to avoid modifying the original
        comparison_data = qb_data.copy()
        
        # Ensure required traditional metrics exist
        traditional_metrics = ['passer_rating', 'qbr']
        for metric in traditional_metrics:
            if metric not in comparison_data.columns:
                comparison_data[metric] = np.nan
        
        # Create rankings for each metric
        if len(comparison_data) > 0:
            # Rank by composite score (descending)
            comparison_data['composite_rank'] = comparison_data['qb_composite_score'].rank(ascending=False)
            
            # Rank by traditional metrics if available
            for metric in traditional_metrics:
                if not comparison_data[metric].isna().all():
                    comparison_data[f'{metric}_rank'] = comparison_data[metric].rank(ascending=False)
            
            # Calculate differences in rankings
            for metric in traditional_metrics:
                rank_col = f'{metric}_rank'
                if rank_col in comparison_data.columns:
                    comparison_data[f'diff_from_{metric}'] = comparison_data['composite_rank'] - comparison_data[rank_col]
        
        logger.info("Compared custom QB metric with traditional rating systems")
        return comparison_data

    def identify_over_underrated_qbs(self, qb_data: pd.DataFrame, threshold: float = 5.0) -> Dict[str, List[str]]:
        """
        Identify QBs who may be over or underrated by traditional metrics.
        
        Args:
            qb_data: DataFrame with QB statistics and ranking comparisons
            threshold: Ranking difference threshold to consider a QB over/underrated

        Returns:
            Dictionary with lists of over and underrated QBs
        """
        # Filter for QBs with significant ranking differences
        overrated = []
        underrated = []
        
        # Check differences against passer rating
        if 'diff_from_passer_rating' in qb_data.columns:
            # Negative difference means ranked higher in traditional metric than composite (overrated)
            overrated_qbs = qb_data[qb_data['diff_from_passer_rating'] < -threshold]
            overrated_qbs = overrated_qbs.sort_values('diff_from_passer_rating')
            
            # Positive difference means ranked lower in traditional metric than composite (underrated)
            underrated_qbs = qb_data[qb_data['diff_from_passer_rating'] > threshold]
            underrated_qbs = underrated_qbs.sort_values('diff_from_passer_rating', ascending=False)
            
            # Create lists of QB names
            if 'player_name' in qb_data.columns:
                overrated = overrated_qbs['player_name'].tolist()
                underrated = underrated_qbs['player_name'].tolist()
        
        logger.info(f"Identified {len(overrated)} overrated and {len(underrated)} underrated QBs")
        return {
            'overrated': overrated,
            'underrated': underrated
        }

    def save_metric_results(self, qb_data: pd.DataFrame, season: int = 2023) -> None:
        """
        Save the metric results to CSV file.

        Args:
            qb_data: DataFrame with QB metrics
            season: NFL season year
        """
        metrics_dir = os.path.join(self.data_dir, f'season_{season}', 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        file_path = os.path.join(metrics_dir, 'qb_composite_metrics.csv')
        qb_data.to_csv(file_path, index=False)
        logger.info(f"Saved QB metric results to {file_path}")

    def develop_qb_metric(self, qb_data: Optional[pd.DataFrame] = None, season: int = 2023) -> pd.DataFrame:
        """
        Develop and calculate the complete QB performance metric.
        
        This is the main method that orchestrates the entire metric development process.

        Args:
            qb_data: Optional DataFrame with QB data (if None, will load from file)
            season: NFL season year

        Returns:
            DataFrame with calculated QB metrics
        """
        try:
            # Load data if not provided
            if qb_data is None:
                qb_data = self.load_processed_data(season)
            
            # Calculate the composite QB metric
            metric_data = self.calculate_composite_qb_metric(qb_data)
            
            # Compare with traditional metrics
            comparison_data = self.compare_with_traditional_metrics(metric_data)
            
            # Identify over/underrated QBs
            self.identify_over_underrated_qbs(comparison_data)
            
            # Save results
            self.save_metric_results(comparison_data, season)
            
            logger.info("Successfully developed and calculated QB performance metric")
            return comparison_data
            
        except Exception as e:
            logger.error(f"Error developing QB metric: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../data_processing')
    from data_processor import NFLDataProcessor
    
    # 1. Load processed data
    processor = NFLDataProcessor(data_dir='../../data')
    processed_data = processor.load_raw_data(season=2023)
    qb_data = processor.process_qb_data(processed_data)
    
    # 2. Develop QB metric
    metric_developer = QBMetricDeveloper(data_dir='../../data')
    qb_metrics = metric_developer.develop_qb_metric(qb_data)
    
    # 3. Display top QBs by composite score
    top_qbs = qb_metrics.sort_values('qb_composite_score', ascending=False).head(10)
    print("\nTop 10 QBs by Composite Score:")
    for idx, qb in top_qbs.iterrows():
        print(f"{qb['player_name']}: {qb['qb_composite_score']}") 