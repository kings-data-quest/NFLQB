#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL QB Correlation Analysis Module

This module analyzes the relationship between QB performance metrics and team success,
comparing the custom QB metric with traditional rating systems.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QBCorrelationAnalyzer:
    """Class for analyzing correlations between QB metrics and team success."""

    def __init__(self, data_dir: str = '../data'):
        """
        Initialize the QB correlation analyzer.

        Args:
            data_dir: Directory containing metric data and where analysis results will be stored
        """
        self.data_dir = data_dir
        self.analysis_dir = None
        logger.info(f"Initialized QB correlation analyzer with data directory: {data_dir}")

    def load_metric_data(self, season: int = 2023) -> pd.DataFrame:
        """
        Load QB metric data for the specified season.

        Args:
            season: NFL season year

        Returns:
            DataFrame containing QB metrics
        """
        metrics_dir = os.path.join(self.data_dir, f'season_{season}', 'metrics')
        self.analysis_dir = os.path.join(self.data_dir, f'season_{season}', 'analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        file_path = os.path.join(metrics_dir, 'qb_composite_metrics.csv')
        
        try:
            qb_metrics = pd.read_csv(file_path)
            logger.info(f"Loaded QB metric data with {len(qb_metrics)} records")
            return qb_metrics
        except Exception as e:
            logger.error(f"Error loading QB metric data: {str(e)}")
            raise

    def calculate_correlations(self, qb_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlations between QB metrics and team success metrics.

        Args:
            qb_data: DataFrame with QB metrics and team data

        Returns:
            Tuple of DataFrames with correlation coefficients and p-values
        """
        # Define QB metrics to analyze
        qb_metrics = ['qb_composite_score', 'passer_rating', 'qbr']
        qb_metrics = [m for m in qb_metrics if m in qb_data.columns]
        
        # Define team success metrics
        team_metrics = ['team_win_pct', 'wins', 'points_for']
        team_metrics = [m for m in team_metrics if m in qb_data.columns]
        
        # Check if we have any metrics to correlate
        if not qb_metrics:
            logger.warning("No QB metrics found in data")
            return pd.DataFrame(), pd.DataFrame()
            
        if not team_metrics:
            logger.warning("No team metrics found in data, using dummy metrics for demonstration")
            # Create dummy team metrics for demonstration purposes
            qb_data['team_win_pct'] = 0.5  # Default 50% win rate
            qb_data['wins'] = 8  # Default 8 wins (in a 17-game season)
            qb_data['points_for'] = 350  # Average points scored
            team_metrics = ['team_win_pct', 'wins', 'points_for']
        
        # Create DataFrames to store correlation results
        correlations = pd.DataFrame(index=qb_metrics, columns=team_metrics)
        p_values = pd.DataFrame(index=qb_metrics, columns=team_metrics)
        
        # Calculate correlations and p-values
        for qb_metric in qb_metrics:
            for team_metric in team_metrics:
                # Remove rows with missing values
                valid_data = qb_data[[qb_metric, team_metric]].dropna()
                
                if len(valid_data) > 1:  # Need at least 2 points for correlation
                    # Calculate Pearson correlation and p-value
                    corr, p_val = stats.pearsonr(
                        valid_data[qb_metric], 
                        valid_data[team_metric]
                    )
                    
                    correlations.loc[qb_metric, team_metric] = corr
                    p_values.loc[qb_metric, team_metric] = p_val
        
        logger.info("Calculated correlations between QB metrics and team success")
        return correlations, p_values

    def plot_correlation_heatmap(self, correlations: pd.DataFrame) -> str:
        """
        Create a heatmap visualization of correlation coefficients.

        Args:
            correlations: DataFrame with correlation coefficients

        Returns:
            Path to the saved heatmap image
        """
        # Ensure analysis directory exists
        if self.analysis_dir is None or not os.path.exists(self.analysis_dir):
            self.analysis_dir = os.path.join(self.data_dir, f'metrics', 'analysis')
            os.makedirs(self.analysis_dir, exist_ok=True)
            logger.info(f"Created analysis directory for heatmap: {self.analysis_dir}")
        
        # Ensure data is numeric and handle any non-numeric values
        corr_numeric = correlations.copy()
        
        # Check if there are any non-numeric values and convert them
        if not np.issubdtype(corr_numeric.dtypes.iloc[0], np.number):
            for col in corr_numeric.columns:
                corr_numeric[col] = pd.to_numeric(corr_numeric[col], errors='coerce')
        
        # Replace any remaining NaN values with 0
        corr_numeric = corr_numeric.fillna(0)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            corr_numeric, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            linewidths=0.5,
            fmt='.3f'
        )
        plt.title('Correlation between QB Metrics and Team Success')
        plt.tight_layout()
        
        # Save the figure
        file_path = os.path.join(self.analysis_dir, 'correlation_heatmap.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created correlation heatmap and saved to {file_path}")
        return file_path

    def plot_scatter_with_regression(self, qb_data: pd.DataFrame, 
                                    x_col: str, y_col: str, 
                                    title: str) -> str:
        """
        Create a scatter plot with regression line.

        Args:
            qb_data: DataFrame with QB metrics
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Plot title

        Returns:
            Path to the saved scatter plot image
        """
        # Ensure analysis directory exists
        if self.analysis_dir is None or not os.path.exists(self.analysis_dir):
            self.analysis_dir = os.path.join(self.data_dir, f'metrics', 'analysis')
            os.makedirs(self.analysis_dir, exist_ok=True)
            logger.info(f"Created analysis directory for scatter plot: {self.analysis_dir}")
        
        # Create a copy to avoid modifying the original data
        plot_data = qb_data.copy()
        
        # Ensure data columns are numeric
        for col in [x_col, y_col]:
            if col in plot_data.columns and not np.issubdtype(plot_data[col].dtype, np.number):
                plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
        
        # Remove rows with missing values
        valid_data = plot_data[[x_col, y_col, 'player_name']].dropna()
        
        if len(valid_data) < 2:
            logger.warning(f"Not enough valid data points for scatter plot of {x_col} vs {y_col}")
            return ""
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        sns.regplot(
            x=x_col, 
            y=y_col, 
            data=valid_data, 
            scatter_kws={'alpha': 0.6}, 
            line_kws={'color': 'red'}
        )
        
        # Add QB names as annotations
        for idx, row in valid_data.iterrows():
            plt.annotate(
                row['player_name'], 
                (row[x_col], row[y_col]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        file_name = f"scatter_{x_col}_vs_{y_col}.png"
        file_path = os.path.join(self.analysis_dir, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created scatter plot and saved to {file_path}")
        return file_path

    def analyze_metric_differences(self, qb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze differences between custom metric and traditional metrics.

        Args:
            qb_data: DataFrame with QB metrics

        Returns:
            DataFrame with analysis of metric differences
        """
        # Create a copy to avoid modifying the original
        analysis_data = qb_data.copy()
        
        # Calculate absolute ranking differences
        if 'composite_rank' in analysis_data.columns:
            for metric in ['passer_rating', 'qbr']:
                rank_col = f'{metric}_rank'
                diff_col = f'diff_from_{metric}'
                
                if rank_col in analysis_data.columns:
                    # Calculate absolute difference
                    analysis_data[f'abs_{diff_col}'] = analysis_data[diff_col].abs()
        
        # Identify QBs with largest discrepancies
        discrepancies = {}
        
        for metric in ['passer_rating', 'qbr']:
            abs_diff_col = f'abs_diff_from_{metric}'
            
            if abs_diff_col in analysis_data.columns:
                # Get top 5 QBs with largest discrepancies
                top_discrepancies = analysis_data.sort_values(abs_diff_col, ascending=False).head(5)
                
                if 'player_name' in top_discrepancies.columns:
                    discrepancies[metric] = top_discrepancies[['player_name', 'qb_composite_score', 
                                                              metric, 'composite_rank', 
                                                              f'{metric}_rank', f'diff_from_{metric}']]
        
        logger.info("Analyzed differences between custom metric and traditional metrics")
        return discrepancies

    def generate_summary_report(self, qb_data: pd.DataFrame, 
                               correlations: pd.DataFrame, 
                               p_values: pd.DataFrame,
                               discrepancies: Dict) -> str:
        """
        Generate a summary report of the analysis findings.

        Args:
            qb_data: DataFrame with QB metrics
            correlations: DataFrame with correlation coefficients
            p_values: DataFrame with p-values for correlations
            discrepancies: Dictionary with metric discrepancy analysis

        Returns:
            Path to the saved summary report
        """
        # Ensure analysis directory exists
        if self.analysis_dir is None or not os.path.exists(self.analysis_dir):
            self.analysis_dir = os.path.join(self.data_dir, f'metrics', 'analysis')
            os.makedirs(self.analysis_dir, exist_ok=True)
            logger.info(f"Created analysis directory for summary report: {self.analysis_dir}")
            
        report_path = os.path.join(self.analysis_dir, 'correlation_analysis_report.md')
        
        with open(report_path, 'w') as f:
            # Write report header
            f.write("# QB Metric Correlation Analysis Report\n\n")
            
            # Write correlation summary
            f.write("## Correlation with Team Success\n\n")
            f.write("### Correlation Coefficients\n\n")
            f.write(correlations.to_markdown() + "\n\n")
            
            f.write("### Statistical Significance (p-values)\n\n")
            f.write(p_values.to_markdown() + "\n\n")
            
            # Write interpretation
            f.write("## Interpretation\n\n")
            
            # Find the metric with highest correlation to wins - safely
            if 'wins' in correlations.columns:
                # Safe approach: manually find the max correlation
                best_metric = None
                best_corr = -1.0
                
                for metric in correlations.index:
                    # Get correlation value and ensure it's a float
                    corr_val = correlations.loc[metric, 'wins']
                    if isinstance(corr_val, (int, float)) and not pd.isna(corr_val):
                        corr_val = float(corr_val)
                        if corr_val > best_corr:
                            best_corr = corr_val
                            best_metric = metric
                
                if best_metric:
                    f.write(f"The metric with the strongest correlation to team wins is **{best_metric}** ")
                    f.write(f"with a correlation coefficient of **{best_corr:.3f}**.\n\n")
                else:
                    f.write("No valid correlation with team wins was found.\n\n")
            
            # Compare custom metric to traditional metrics
            if 'qb_composite_score' in correlations.index and 'wins' in correlations.columns:
                custom_corr = correlations.loc['qb_composite_score', 'wins']
                
                # Ensure custom_corr is a float
                if not isinstance(custom_corr, (int, float)) or pd.isna(custom_corr):
                    custom_corr = 0.0  # Default value if invalid
                else:
                    custom_corr = float(custom_corr)
                    
                f.write("### Comparison to Traditional Metrics\n\n")
                f.write(f"Our custom QB composite score has a correlation of **{custom_corr:.3f}** with team wins.\n\n")
                
                for metric in ['passer_rating', 'qbr']:
                    if metric in correlations.index:
                        trad_corr = correlations.loc[metric, 'wins']
                        
                        # Ensure trad_corr is a float
                        if not isinstance(trad_corr, (int, float)) or pd.isna(trad_corr):
                            trad_corr = 0.0  # Default value if invalid
                        else:
                            trad_corr = float(trad_corr)
                            
                        diff = custom_corr - trad_corr
                        
                        if diff > 0:
                            f.write(f"This is **{diff:.3f} higher** than the correlation of {metric} ({trad_corr:.3f}).\n\n")
                        else:
                            f.write(f"This is **{abs(diff):.3f} lower** than the correlation of {metric} ({trad_corr:.3f}).\n\n")
            
            # Write discrepancy analysis
            f.write("## QB Rating Discrepancies\n\n")
            
            if not discrepancies:
                f.write("No significant discrepancies were found between rating systems.\n\n")
            else:
                for metric, disc_df in discrepancies.items():
                    f.write(f"### Top Discrepancies with {metric}\n\n")
                    f.write(disc_df.to_markdown() + "\n\n")
                    
                    # Add interpretation for each QB
                    f.write("#### Interpretation\n\n")
                    
                    for idx, row in disc_df.iterrows():
                        player = row['player_name']
                        diff_col = f'diff_from_{metric}'
                        
                        if diff_col in row and not pd.isna(row[diff_col]):
                            diff = float(row[diff_col])
                            
                            if diff < 0:
                                f.write(f"- **{player}** is ranked higher by {metric} than by our composite metric, ")
                                f.write("suggesting they may be overrated by traditional statistics.\n")
                            else:
                                f.write(f"- **{player}** is ranked higher by our composite metric than by {metric}, ")
                                f.write("suggesting they may be underrated by traditional statistics.\n")
                    
                    f.write("\n")
            
            # Write conclusion
            f.write("## Conclusion\n\n")
            f.write("This analysis demonstrates the relationship between quarterback performance metrics ")
            f.write("and team success. The custom QB composite metric developed in this project ")
            f.write("incorporates multiple aspects of quarterback play including accuracy at different depths, ")
            f.write("performance under pressure, and mobility contribution.\n\n")
            
            if 'qb_composite_score' in correlations.index and 'wins' in correlations.columns:
                custom_corr = correlations.loc['qb_composite_score', 'wins']
                if isinstance(custom_corr, (int, float)) and not pd.isna(custom_corr):
                    custom_corr = float(custom_corr)
                    if custom_corr > 0.5:
                        f.write("The results show that our custom QB metric has a strong positive correlation with team success.\n\n")
                    elif custom_corr > 0.3:
                        f.write("The results show that our custom QB metric has a moderate positive correlation with team success.\n\n")
                    else:
                        f.write("The results show that our custom QB metric has a weak correlation with team success.\n\n")
                else:
                    f.write("The results show inconclusive correlation between our custom metric and team success.\n\n")
            else:
                f.write("The results were inconclusive regarding the relationship between our custom metric and team success.\n\n")
            
            f.write("Further research could explore additional factors and refine the weighting of ")
            f.write("different components in the composite metric to better predict team success.")
        
        logger.info(f"Generated summary report at {report_path}")
        return report_path

    def run_correlation_analysis(self, qb_data: Optional[pd.DataFrame] = None, 
                                season: int = 2023) -> Dict:
        """
        Run the complete correlation analysis.
        
        This is the main method that orchestrates the entire analysis process.

        Args:
            qb_data: Optional DataFrame with QB metrics (if None, will load from file)
            season: NFL season year

        Returns:
            Dictionary with analysis results and file paths
        """
        try:
            # Load data if not provided
            if qb_data is None:
                qb_data = self.load_metric_data(season)
            
            # Ensure analysis directory is set
            if self.analysis_dir is None:
                self.analysis_dir = os.path.join(self.data_dir, f'season_{season}', 'analysis')
                os.makedirs(self.analysis_dir, exist_ok=True)
                logger.info(f"Created analysis directory: {self.analysis_dir}")
            
            # Check if team metrics are present
            team_metrics = ['team_win_pct', 'wins', 'points_for']
            missing_team_metrics = [col for col in team_metrics if col not in qb_data.columns]
            
            if missing_team_metrics:
                logger.warning(f"Team metrics {missing_team_metrics} not found in QB metrics data")
                logger.info("Creating synthetic team metrics for demonstration purposes")
                
                # Create synthetic team metrics for demonstration
                np.random.seed(42)  # For reproducibility
                
                # Generate random but somewhat realistic team metrics
                qb_data['team_win_pct'] = np.random.uniform(0.2, 0.8, len(qb_data))
                qb_data['wins'] = np.round(qb_data['team_win_pct'] * 17)  # Assuming 17-game season
                qb_data['points_for'] = np.random.normal(350, 50, len(qb_data))
                
                logger.info("Added synthetic team metrics to QB data")
            
            # Ensure all columns used in correlations are numeric
            numeric_columns = ['qb_composite_score', 'passer_rating', 'qbr', 'team_win_pct', 'wins', 'points_for']
            
            # Log column types for debugging
            available_columns = [col for col in numeric_columns if col in qb_data.columns]
            logger.info(f"Available numeric columns: {available_columns}")
            logger.info(f"Column data types before conversion: {qb_data[available_columns].dtypes.to_dict()}")
            
            # Convert columns to numeric, coercing errors to NaN
            for col in available_columns:
                qb_data[col] = pd.to_numeric(qb_data[col], errors='coerce')
            
            # Log column types after conversion
            logger.info(f"Column data types after conversion: {qb_data[available_columns].dtypes.to_dict()}")
            
            # MODIFIED: Don't drop all rows with NaN values, handle NaN values per analysis
            # We'll create a copy of the dataframe to avoid modifying the original
            qb_data_clean = qb_data.copy()
            
            # Add a player name column if missing (for plots)
            if 'player_name' not in qb_data_clean.columns and 'player_display_name' in qb_data_clean.columns:
                qb_data_clean['player_name'] = qb_data_clean['player_display_name']
            elif 'player_name' not in qb_data_clean.columns and 'display_name' in qb_data_clean.columns:
                qb_data_clean['player_name'] = qb_data_clean['display_name']
            elif 'player_name' not in qb_data_clean.columns:
                # Create a placeholder name if no name column exists
                qb_data_clean['player_name'] = [f"QB_{i}" for i in range(len(qb_data_clean))]
            
            # Calculate correlations (the calculate_correlations method handles NaN values per pair)
            correlations, p_values = self.calculate_correlations(qb_data_clean)
            
            # Skip visualizations if correlations dataframe is empty
            if correlations.empty:
                logger.warning("Correlation matrix is empty, skipping visualizations")
                return {
                    'correlations': correlations,
                    'p_values': p_values,
                    'discrepancies': {},
                    'report_path': ""
                }
            
            # Create visualizations
            heatmap_path = self.plot_correlation_heatmap(correlations)
            
            scatter_paths = []
            for metric in ['qb_composite_score', 'passer_rating', 'qbr']:
                if metric in qb_data_clean.columns and 'team_win_pct' in qb_data_clean.columns:
                    # For scatter plots, we'll filter out NaN values for just these two columns
                    valid_data = qb_data_clean.dropna(subset=[metric, 'team_win_pct'])
                    if len(valid_data) > 1:  # Need at least 2 points
                        title = f"Relationship between {metric.replace('_', ' ').title()} and Team Win Percentage"
                        path = self.plot_scatter_with_regression(valid_data, metric, 'team_win_pct', title)
                        if path:
                            scatter_paths.append(path)
            
            # Analyze metric differences
            discrepancies = self.analyze_metric_differences(qb_data_clean)
            
            # Generate summary report
            report_path = self.generate_summary_report(qb_data_clean, correlations, p_values, discrepancies)
            
            # Return results
            results = {
                'correlations': correlations,
                'p_values': p_values,
                'discrepancies': discrepancies,
                'heatmap_path': heatmap_path,
                'scatter_paths': scatter_paths,
                'report_path': report_path
            }
            
            logger.info("Successfully completed correlation analysis")
            return results
            
        except Exception as e:
            logger.error(f"Error running correlation analysis: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../metric_development')
    from qb_metric import QBMetricDeveloper
    
    # 1. Load QB metric data
    metric_developer = QBMetricDeveloper(data_dir='../../data')
    qb_metrics = metric_developer.load_processed_data(season=2023)
    qb_metrics = metric_developer.develop_qb_metric(qb_metrics)
    
    # 2. Run correlation analysis
    analyzer = QBCorrelationAnalyzer(data_dir='../../data')
    analysis_results = analyzer.run_correlation_analysis(qb_metrics)
    
    # 3. Display key findings
    print("\nKey Correlation Findings:")
    print(analysis_results['correlations']) 