#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL QB Visualization Module

This module generates visualizations for QB performance metrics and analysis results.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QBVisualizer:
    """Class for creating visualizations of QB performance metrics."""

    def __init__(self, data_dir: str = '../data'):
        """
        Initialize the QB visualizer.

        Args:
            data_dir: Directory containing data and where visualizations will be stored
        """
        self.data_dir = data_dir
        self.viz_dir = None
        logger.info(f"Initialized QB visualizer with data directory: {data_dir}")

    def load_metric_data(self, season: int = 2023) -> pd.DataFrame:
        """
        Load QB metric data for the specified season.

        Args:
            season: NFL season year

        Returns:
            DataFrame containing QB metrics
        """
        metrics_dir = os.path.join(self.data_dir, f'season_{season}', 'metrics')
        self.viz_dir = os.path.join(self.data_dir, f'season_{season}', 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        file_path = os.path.join(metrics_dir, 'qb_composite_metrics.csv')
        
        try:
            qb_metrics = pd.read_csv(file_path)
            logger.info(f"Loaded QB metric data with {len(qb_metrics)} records")
            return qb_metrics
        except Exception as e:
            logger.error(f"Error loading QB metric data: {str(e)}")
            raise

    def create_top_qbs_bar_chart(self, qb_data: pd.DataFrame, 
                                metric: str = 'qb_composite_score',
                                n_qbs: int = 10) -> str:
        """
        Create a bar chart of top QBs by the specified metric.

        Args:
            qb_data: DataFrame with QB metrics
            metric: Metric to rank QBs by
            n_qbs: Number of top QBs to include

        Returns:
            Path to the saved visualization
        """
        if metric not in qb_data.columns or 'player_name' not in qb_data.columns:
            logger.warning(f"Required columns not found for top QBs bar chart")
            return ""
        
        # Get top QBs
        top_qbs = qb_data.sort_values(metric, ascending=False).head(n_qbs)
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = plt.barh(top_qbs['player_name'], top_qbs[metric])
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center')
        
        # Add title and labels
        plt.title(f'Top {n_qbs} QBs by {metric.replace("_", " ").title()}')
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel('Quarterback')
        plt.tight_layout()
        
        # Save the figure
        file_path = os.path.join(self.viz_dir, f'top_{n_qbs}_qbs_{metric}.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created top QBs bar chart and saved to {file_path}")
        return file_path

    def create_metric_components_radar_chart(self, qb_data: pd.DataFrame, 
                                           qb_names: List[str],
                                           components: List[str]) -> str:
        """
        Create a radar chart comparing QB metric components.

        Args:
            qb_data: DataFrame with QB metrics
            qb_names: List of QB names to include
            components: List of metric components to compare

        Returns:
            Path to the saved visualization
        """
        if 'player_name' not in qb_data.columns:
            logger.warning("Player name column not found for radar chart")
            return ""
        
        # Filter for selected QBs
        selected_qbs = qb_data[qb_data['player_name'].isin(qb_names)]
        
        if len(selected_qbs) == 0:
            logger.warning("No matching QBs found for radar chart")
            return ""
        
        # Check if all components exist
        valid_components = [c for c in components if c in selected_qbs.columns]
        
        if len(valid_components) == 0:
            logger.warning("No valid metric components found for radar chart")
            return ""
        
        # Create radar chart using plotly
        fig = go.Figure()
        
        for _, qb in selected_qbs.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[qb[c] for c in valid_components],
                theta=valid_components,
                fill='toself',
                name=qb['player_name']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=f"QB Metric Components Comparison",
            showlegend=True
        )
        
        # Save the figure
        file_path = os.path.join(self.viz_dir, 'qb_radar_comparison.html')
        fig.write_html(file_path)
        
        # Also save as image
        img_path = os.path.join(self.viz_dir, 'qb_radar_comparison.png')
        fig.write_image(img_path, width=1000, height=800)
        
        logger.info(f"Created QB radar chart and saved to {file_path}")
        return file_path

    def create_metric_correlation_matrix(self, qb_data: pd.DataFrame) -> str:
        """
        Create a correlation matrix heatmap for QB metrics.

        Args:
            qb_data: DataFrame with QB metrics

        Returns:
            Path to the saved visualization
        """
        # Select numeric columns only
        numeric_data = qb_data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=False, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            linewidths=0.5
        )
        plt.title('Correlation Matrix of QB Metrics')
        plt.tight_layout()
        
        # Save the figure
        file_path = os.path.join(self.viz_dir, 'metric_correlation_matrix.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created metric correlation matrix and saved to {file_path}")
        return file_path

    def create_metric_distribution_plots(self, qb_data: pd.DataFrame, 
                                        metrics: List[str]) -> List[str]:
        """
        Create distribution plots for QB metrics.

        Args:
            qb_data: DataFrame with QB metrics
            metrics: List of metrics to visualize

        Returns:
            List of paths to the saved visualizations
        """
        file_paths = []
        
        for metric in metrics:
            if metric in qb_data.columns:
                plt.figure(figsize=(10, 6))
                
                # Create distribution plot
                sns.histplot(qb_data[metric], kde=True)
                
                # Add mean and median lines
                plt.axvline(qb_data[metric].mean(), color='red', linestyle='--', 
                           label=f'Mean: {qb_data[metric].mean():.2f}')
                plt.axvline(qb_data[metric].median(), color='green', linestyle='-', 
                           label=f'Median: {qb_data[metric].median():.2f}')
                
                # Add title and labels
                plt.title(f'Distribution of {metric.replace("_", " ").title()}')
                plt.xlabel(metric.replace('_', ' ').title())
                plt.ylabel('Frequency')
                plt.legend()
                plt.tight_layout()
                
                # Save the figure
                file_path = os.path.join(self.viz_dir, f'{metric}_distribution.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                file_paths.append(file_path)
                logger.info(f"Created distribution plot for {metric} and saved to {file_path}")
        
        return file_paths

    def create_metric_vs_wins_scatter(self, qb_data: pd.DataFrame, 
                                     metrics: List[str]) -> List[str]:
        """
        Create scatter plots of QB metrics vs team wins.

        Args:
            qb_data: DataFrame with QB metrics
            metrics: List of metrics to plot against wins

        Returns:
            List of paths to the saved visualizations
        """
        file_paths = []
        
        if 'wins' not in qb_data.columns or 'player_name' not in qb_data.columns:
            logger.warning("Required columns not found for metric vs wins scatter plots")
            return file_paths
        
        for metric in metrics:
            if metric in qb_data.columns:
                plt.figure(figsize=(12, 8))
                
                # Create scatter plot
                sns.regplot(
                    x=metric, 
                    y='wins', 
                    data=qb_data, 
                    scatter_kws={'alpha': 0.6}, 
                    line_kws={'color': 'red'}
                )
                
                # Add QB names as annotations
                for idx, row in qb_data.iterrows():
                    plt.annotate(
                        row['player_name'], 
                        (row[metric], row['wins']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8
                    )
                
                # Calculate and display correlation
                corr = qb_data[[metric, 'wins']].corr().iloc[0, 1]
                plt.annotate(
                    f'Correlation: {corr:.3f}',
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
                
                # Add title and labels
                plt.title(f'Relationship between {metric.replace("_", " ").title()} and Team Wins')
                plt.xlabel(metric.replace('_', ' ').title())
                plt.ylabel('Team Wins')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save the figure
                file_path = os.path.join(self.viz_dir, f'{metric}_vs_wins.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                file_paths.append(file_path)
                logger.info(f"Created scatter plot for {metric} vs wins and saved to {file_path}")
        
        return file_paths

    def create_interactive_dashboard(self, qb_data: pd.DataFrame) -> str:
        """
        Create an interactive dashboard with multiple QB visualizations.

        Args:
            qb_data: DataFrame with QB metrics

        Returns:
            Path to the saved dashboard
        """
        # Check if required columns exist
        required_cols = ['qb_composite_score', 'wins']
        
        if not all(col in qb_data.columns for col in required_cols):
            logger.warning("Required columns not found for interactive dashboard")
            return ""
        
        # Create a player name column if it doesn't exist
        if 'player_name' not in qb_data.columns:
            # Try to use display_name if available
            if 'display_name' in qb_data.columns:
                qb_data['player_name'] = qb_data['display_name']
            else:
                # Create default names
                qb_data['player_name'] = [f"QB_{i+1}" for i in range(len(qb_data))]
            logger.info("Created player_name column for dashboard")
        
        # Create the dashboard with subplots
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                'QB Composite Score vs Team Wins',
                'Top 10 QBs by Composite Score',
                'Completion % by Pass Depth',
                'Regular vs Pressure Completion %'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. QB Score vs Team Wins scatter plot
        fig.add_trace(
            go.Scatter(
                x=qb_data['qb_composite_score'],
                y=qb_data['wins'],
                mode='markers+text',
                text=qb_data['player_name'],
                textposition='top center',
                marker=dict(
                    size=10,
                    color=qb_data['qb_composite_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='QB Score')
                ),
                name='QB Score vs Wins'
            ),
            row=1, col=1
        )
        
        # Add trendline
        if len(qb_data) > 1:  # Need at least 2 points for regression
            z = np.polyfit(qb_data['qb_composite_score'], qb_data['wins'], 1)
            p = np.poly1d(z)
            
            x_range = np.linspace(qb_data['qb_composite_score'].min(), qb_data['qb_composite_score'].max(), 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                ),
                row=1, col=1
            )
        
        # 2. Top 10 QBs bar chart
        top_qbs = qb_data.sort_values('qb_composite_score', ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(
                x=top_qbs['player_name'],
                y=top_qbs['qb_composite_score'],
                marker=dict(color='rgba(50, 171, 96, 0.7)'),
                name='Top 10 QBs'
            ),
            row=1, col=2
        )
        
        # 3. Completion % by Pass Depth
        depth_cols = ['behind_los_comp_pct', 'short_comp_pct', 'medium_comp_pct', 'deep_comp_pct']
        if all(col in qb_data.columns for col in depth_cols):
            # Get top 5 QBs by composite score
            top5_qbs = qb_data.sort_values('qb_composite_score', ascending=False).head(5)
            
            for idx, qb in top5_qbs.iterrows():
                fig.add_trace(
                    go.Bar(
                        x=['Behind LOS', 'Short', 'Medium', 'Deep'],
                        y=[qb[col] for col in depth_cols],
                        name=qb['player_name']
                    ),
                    row=2, col=1
                )
        
        # 4. Regular vs Pressure Completion %
        if 'completions' in qb_data.columns and 'attempts' in qb_data.columns and 'pressure_comp_pct' in qb_data.columns:
            # Calculate regular completion percentage
            qb_data['regular_comp_pct'] = (qb_data['completions'] / qb_data['attempts']) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=qb_data['regular_comp_pct'],
                    y=qb_data['pressure_comp_pct'],
                    mode='markers+text',
                    text=qb_data['player_name'],
                    textposition='top center',
                    name='QBs',
                    marker=dict(size=10, opacity=0.7)
                ),
                row=2, col=2
            )
            
            # Add diagonal line (y=x)
            max_val = max(qb_data['regular_comp_pct'].max(), qb_data['pressure_comp_pct'].max())
            min_val = min(qb_data['regular_comp_pct'].min(), qb_data['pressure_comp_pct'].min())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Equal Performance',
                    line=dict(color='black', dash='dash')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='NFL QB Performance Dashboard',
            height=900,
            width=1200,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='QB Composite Score', row=1, col=1)
        fig.update_yaxes(title_text='Team Wins', row=1, col=1)
        
        fig.update_xaxes(title_text='Quarterback', row=1, col=2)
        fig.update_yaxes(title_text='Composite Score', row=1, col=2)
        
        fig.update_xaxes(title_text='Pass Depth', row=2, col=1)
        fig.update_yaxes(title_text='Completion Percentage', row=2, col=1)
        
        fig.update_xaxes(title_text='Regular Completion %', row=2, col=2)
        fig.update_yaxes(title_text='Under Pressure Completion %', row=2, col=2)
        
        # Save the dashboard
        file_path = os.path.join(self.viz_dir, 'qb_performance_dashboard.html')
        fig.write_html(file_path)
        
        logger.info(f"Created interactive dashboard and saved to {file_path}")
        return file_path

    def create_visualizations(self, qb_data: Optional[pd.DataFrame] = None, 
                             season: int = 2023) -> Dict[str, List[str]]:
        """
        Create all visualizations for QB analysis.
        
        This is the main method that orchestrates the entire visualization process.

        Args:
            qb_data: Optional DataFrame with QB metrics (if None, will load from file)
            season: NFL season year

        Returns:
            Dictionary with paths to all created visualizations
        """
        try:
            # Ensure visualization directory exists
            if self.viz_dir is None:
                self.viz_dir = os.path.join(self.data_dir, f'season_{season}', 'visualizations')
                os.makedirs(self.viz_dir, exist_ok=True)
                logger.info(f"Created visualization directory: {self.viz_dir}")
            
            # Load data if not provided
            if qb_data is None:
                qb_data = self.load_metric_data(season)
            
            # Create visualizations
            viz_paths = {}
            
            # 1. Top QBs bar chart
            top_qbs_path = self.create_top_qbs_bar_chart(qb_data)
            if top_qbs_path:
                viz_paths['top_qbs'] = [top_qbs_path]
            
            # 2. Metric components radar chart
            # Select top 5 QBs by composite score
            if 'qb_composite_score' in qb_data.columns and 'player_name' in qb_data.columns:
                top_qbs = qb_data.sort_values('qb_composite_score', ascending=False).head(5)
                qb_names = top_qbs['player_name'].tolist()
                
                components = [
                    'depth_adjusted_comp_pct_normalized',
                    'pressure_performance_normalized',
                    'mobility_contribution_normalized',
                    'yards_per_attempt_normalized',
                    'td_int_ratio_normalized',
                    'sack_avoidance_normalized'
                ]
                
                radar_path = self.create_metric_components_radar_chart(qb_data, qb_names, components)
                if radar_path:
                    viz_paths['radar_chart'] = [radar_path]
            
            # 3. Correlation matrix
            corr_matrix_path = self.create_metric_correlation_matrix(qb_data)
            if corr_matrix_path:
                viz_paths['correlation_matrix'] = [corr_matrix_path]
            
            # 4. Distribution plots
            dist_metrics = ['qb_composite_score', 'passer_rating', 'depth_adjusted_comp_pct', 'pressure_performance']
            dist_paths = self.create_metric_distribution_plots(qb_data, dist_metrics)
            if dist_paths:
                viz_paths['distributions'] = [path for path in dist_paths if path]
            
            # 5. Metric vs wins scatter plots
            scatter_metrics = ['qb_composite_score', 'passer_rating', 'qbr']
            scatter_paths = self.create_metric_vs_wins_scatter(qb_data, scatter_metrics)
            if scatter_paths:
                viz_paths['scatter_plots'] = [path for path in scatter_paths if path]
            
            # 6. Interactive dashboard
            dashboard_path = self.create_interactive_dashboard(qb_data)
            if dashboard_path:
                viz_paths['dashboard'] = [dashboard_path]
            
            logger.info(f"Successfully created all visualizations")
            return viz_paths
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
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
    
    # 2. Create visualizations
    visualizer = QBVisualizer(data_dir='../../data')
    viz_paths = visualizer.create_visualizations(qb_metrics)
    
    # 3. Display paths to created visualizations
    for viz_type, paths in viz_paths.items():
        print(f"\n{viz_type.replace('_', ' ').title()}:")
        for path in paths:
            if path:
                print(f"  - {path}") 