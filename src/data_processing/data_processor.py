#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL QB Data Processor Module

This module handles cleaning, transforming, and preparing NFL QB data for analysis.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLDataProcessor:
    """Class for processing NFL data for QB analysis."""

    def __init__(self, data_dir: str = '../data', season: int = 2023):
        """
        Initialize the NFL data processor.

        Args:
            data_dir: Directory containing the raw data and where processed data will be stored
            season: NFL season year for default output directory
        """
        self.data_dir = data_dir
        self.season = season
        self.processed_dir = os.path.join(self.data_dir, f'season_{season}', 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        logger.info(f"Initialized NFL data processor with data directory: {data_dir}")

    def load_raw_data(self, season: int = None) -> Dict[str, pd.DataFrame]:
        """
        Load raw data files for the specified season.

        Args:
            season: NFL season year (defaults to self.season if None)

        Returns:
            Dictionary containing different datasets
        """
        if season is None:
            season = self.season
            
        season_dir = os.path.join(self.data_dir, f'season_{season}', 'raw')
        self.processed_dir = os.path.join(self.data_dir, f'season_{season}', 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        
        datasets = {}
        
        # List all CSV files in the raw data directory
        try:
            files = [f for f in os.listdir(season_dir) if f.endswith('.csv')]
            
            for file in files:
                name = file.replace('.csv', '')
                file_path = os.path.join(season_dir, file)
                datasets[name] = pd.read_csv(file_path)
                logger.info(f"Loaded {name} data with {len(datasets[name])} records")
                
            return datasets
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise

    def filter_qb_data(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Filter player stats to include only QBs with sufficient play time.

        Args:
            player_stats: DataFrame containing player statistical data

        Returns:
            DataFrame with QB-only data
        """
        # Filter for QBs only
        qb_stats = player_stats[player_stats['position'] == 'QB'].copy()
        
        # Filter for QBs with meaningful playing time (at least 100 passing attempts)
        qb_stats = qb_stats[qb_stats['attempts'] >= 100].copy()
        
        logger.info(f"Filtered data to {len(qb_stats)} QBs with at least 100 passing attempts")
        return qb_stats

    def prepare_pbp_qb_data(self, pbp_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare play-by-play data for QB analysis.

        Args:
            pbp_data: Raw play-by-play data

        Returns:
            Filtered and enriched play-by-play data for QB analysis
        """
        # Check if pbp_data is empty
        if pbp_data.empty:
            logger.warning("Play-by-play data is empty, returning empty DataFrame")
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=[
                'play_type', 'passer_player_id', 'complete_pass', 'touchdown', 'interception',
                'air_yards', 'is_complete', 'is_touchdown', 'is_interception', 'pass_depth',
                'under_pressure', 'passing_yards'
            ])
            
        # Filter for pass plays
        pass_plays = pbp_data[
            (pbp_data['play_type'] == 'pass') &
            (~pbp_data['passer_player_id'].isna())
        ].copy()
        
        # Add useful features
        pass_plays['is_complete'] = pass_plays['complete_pass'] == 1
        pass_plays['is_touchdown'] = pass_plays['touchdown'] == 1
        pass_plays['is_interception'] = pass_plays['interception'] == 1
        
        # Categorize pass depth
        conditions = [
            (pass_plays['air_yards'] < 0),
            (pass_plays['air_yards'] >= 0) & (pass_plays['air_yards'] < 10),
            (pass_plays['air_yards'] >= 10) & (pass_plays['air_yards'] < 20),
            (pass_plays['air_yards'] >= 20)
        ]
        choices = ['behind_los', 'short', 'medium', 'deep']
        pass_plays['pass_depth'] = np.select(conditions, choices, default='unknown')
        
        # Flag for pressure - check for existence of each column
        logger.info(f"PBP columns related to pressure: {[col for col in pass_plays.columns if 'pressure' in col.lower() or 'hit' in col.lower() or 'hurry' in col.lower() or 'sack' in col.lower()]}")
        
        # Create under_pressure field based on available data
        pressure_conditions = []
        
        if 'qb_hit' in pass_plays.columns:
            pressure_conditions.append(pass_plays['qb_hit'] == 1)
            logger.info("Using 'qb_hit' for pressure detection")
            
        if 'sack' in pass_plays.columns:
            pressure_conditions.append(pass_plays['sack'] == 1)
            logger.info("Using 'sack' for pressure detection")
            
        if 'hurry' in pass_plays.columns:
            pressure_conditions.append(pass_plays['hurry'] == 1)
            logger.info("Using 'hurry' for pressure detection")
            
        # If we have any pressure conditions, use them
        if pressure_conditions:
            # Combine conditions with OR (|)
            combined_condition = pressure_conditions[0]
            for condition in pressure_conditions[1:]:
                combined_condition = combined_condition | condition
            pass_plays['under_pressure'] = combined_condition
            logger.info(f"Created 'under_pressure' field using {len(pressure_conditions)} indicators")
        else:
            # If no pressure columns available, default to False
            pass_plays['under_pressure'] = False
            logger.warning("No pressure indicators found in data, all plays marked as not under pressure")
            
        logger.info(f"Prepared {len(pass_plays)} pass plays for QB analysis")
        return pass_plays

    def enrich_qb_stats(
        self, 
        qb_stats: pd.DataFrame, 
        play_by_play: pd.DataFrame, 
        player_info: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Enrich QB statistics with advanced metrics from play-by-play data.

        Args:
            qb_stats: DataFrame with basic QB statistics
            play_by_play: DataFrame with prepared play-by-play data
            player_info: Optional DataFrame with player information

        Returns:
            DataFrame with enriched QB statistics
        """
        # Create a copy to avoid modifying the original
        enhanced_stats = qb_stats.copy()
        
        # Add additional team information if available
        if player_info is not None and 'team_abbr' in player_info.columns and 'player_id' in enhanced_stats.columns:
            logger.info("Adding team information from player_info")
            
            # Get the latest team for each player
            if 'player_id' in player_info.columns:
                team_info = player_info[['player_id', 'team_abbr']].dropna()
                enhanced_stats = enhanced_stats.merge(
                    team_info,
                    on='player_id',
                    how='left'
                )
                logger.info(f"Added team information to {enhanced_stats['team_abbr'].notna().sum()} QB records")
        
        # Check if play-by-play data is empty
        if play_by_play.empty:
            logger.warning("Play-by-play data is empty, skipping advanced metrics calculation")
            # Initialize columns with default values
            for depth in ['behind_los', 'short', 'medium', 'deep']:
                enhanced_stats[f'{depth}_attempts'] = 0
                enhanced_stats[f'{depth}_completions'] = 0
                enhanced_stats[f'{depth}_yards'] = 0
                enhanced_stats[f'{depth}_tds'] = 0
                enhanced_stats[f'{depth}_ints'] = 0
                enhanced_stats[f'{depth}_comp_pct'] = 0.0
            
            # Initialize pressure metrics
            enhanced_stats['pressure_rate'] = 0.0
            enhanced_stats['pressure_comp_pct'] = 0.0
            enhanced_stats['clean_comp_pct'] = 0.0
            enhanced_stats['pressure_comp_diff'] = 0.0
            
            return enhanced_stats
            
        # Group plays by QB
        if 'passer_player_id' in play_by_play.columns and 'player_id' in enhanced_stats.columns:
            logger.info("Calculating advanced QB metrics from play-by-play data")
            
            # Calculate basic passing metrics
            for depth in ['behind_los', 'short', 'medium', 'deep']:
                depth_plays = play_by_play[play_by_play['pass_depth'] == depth]
                
                for qb_id in enhanced_stats['player_id']:
                    qb_plays = depth_plays[depth_plays['passer_player_id'] == qb_id]
                    
                    if len(qb_plays) > 0:
                        # Calculate metrics for this depth
                        attempts = len(qb_plays)
                        completions = qb_plays['is_complete'].sum()
                        yards = qb_plays['passing_yards'].sum()
                        tds = qb_plays['is_touchdown'].sum()
                        ints = qb_plays['is_interception'].sum()
                        
                        # Add to enhanced stats
                        enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, f'{depth}_attempts'] = attempts
                        enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, f'{depth}_completions'] = completions
                        enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, f'{depth}_yards'] = yards
                        enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, f'{depth}_tds'] = tds
                        enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, f'{depth}_ints'] = ints
                        if attempts > 0:
                            enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, f'{depth}_comp_pct'] = (completions / attempts) * 100
                        else:
                            enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, f'{depth}_comp_pct'] = 0
            
            # Add pressure metrics
            for qb_id in enhanced_stats['player_id']:
                qb_plays = play_by_play[play_by_play['passer_player_id'] == qb_id]
                
                if len(qb_plays) > 0:
                    # Total plays under pressure
                    pressure_plays = qb_plays[qb_plays['under_pressure'] == True]
                    clean_plays = qb_plays[qb_plays['under_pressure'] == False]
                    
                    # Calculate metrics under pressure
                    enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, 'pressure_rate'] = (
                        len(pressure_plays) / len(qb_plays) * 100 if len(qb_plays) > 0 else 0
                    )
                    
                    # Pressure vs clean pocket completion %
                    enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, 'pressure_comp_pct'] = (
                        pressure_plays['is_complete'].sum() / len(pressure_plays) * 100 if len(pressure_plays) > 0 else 0
                    )
                    enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, 'clean_comp_pct'] = (
                        clean_plays['is_complete'].sum() / len(clean_plays) * 100 if len(clean_plays) > 0 else 0
                    )
                    
                    # Difference in completion % between clean and pressure
                    enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, 'pressure_comp_diff'] = (
                        enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, 'clean_comp_pct'].values[0] -
                        enhanced_stats.loc[enhanced_stats['player_id'] == qb_id, 'pressure_comp_pct'].values[0]
                    )
        
        logger.info(f"Enhanced QB stats with depth-based and pressure metrics for {len(enhanced_stats)} QBs")
        return enhanced_stats

    def add_team_context(self, qb_stats: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Add team context to QB statistics.

        Args:
            qb_stats: DataFrame with QB statistics
            team_stats: DataFrame with team statistics

        Returns:
            DataFrame with QB statistics enriched with team context
        """
        # Create a copy to avoid modifying the original
        enriched_stats = qb_stats.copy()
        
        # Check if team_stats has the required columns
        required_columns = ['team_abbr', 'wins', 'losses', 'ties', 'points_for', 'points_against']
        missing_columns = [col for col in required_columns if col not in team_stats.columns]
        
        if missing_columns:
            logger.warning(f"Team stats missing columns: {missing_columns}. Skipping team context enrichment.")
            return enriched_stats
        
        # Map team abbreviations to QB records
        team_mapping = team_stats[required_columns].copy()
        
        # Log the team mapping for debugging
        logger.info(f"Team mapping sample: {team_mapping.head().to_dict('records')}")
        
        # Check if 'recent_team' exists in qb_stats
        if 'recent_team' not in enriched_stats.columns:
            logger.warning("QB stats missing 'recent_team' column, trying to use 'team' or 'team_abbr' instead")
            
            # Try alternative column names
            if 'team' in enriched_stats.columns:
                logger.info("Using 'team' column instead of 'recent_team'")
                merge_col = 'team'
            elif 'team_abbr' in enriched_stats.columns:
                logger.info("Using 'team_abbr' column instead of 'recent_team'")
                merge_col = 'team_abbr'
            else:
                logger.warning("No team identifier column found in QB stats, skipping team context enrichment")
                return enriched_stats
        else:
            merge_col = 'recent_team'
        
        # Log QB teams for debugging
        logger.info(f"QB teams: {enriched_stats[merge_col].value_counts().to_dict()}")
        
        # Merge team stats with QB stats
        enriched_stats = enriched_stats.merge(
            team_mapping, 
            left_on=merge_col, 
            right_on='team_abbr', 
            how='left'
        )
        
        # Check for any failed merges
        missing_teams = enriched_stats[enriched_stats['wins'].isna()][merge_col].unique()
        if len(missing_teams) > 0:
            logger.warning(f"Could not find team stats for {len(missing_teams)} teams: {missing_teams}")
        
        # Calculate win percentage
        enriched_stats['team_win_pct'] = (
            enriched_stats['wins'] / 
            (enriched_stats['wins'] + enriched_stats['losses'] + enriched_stats['ties'])
        ).fillna(0)
        
        logger.info(f"Added team context to {len(enriched_stats)} QB records")
        return enriched_stats

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: DataFrame with potentially missing values

        Returns:
            DataFrame with handled missing values
        """
        # Create a copy to avoid modifying the original
        clean_df = df.copy()
        
        # Log the initial number of missing values
        total_missing = clean_df.isna().sum().sum()
        logger.info(f"Handling {total_missing} missing values across {clean_df.shape[1]} columns")
        
        # For numeric columns, use median imputation
        numeric_cols = clean_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            clean_df[numeric_cols] = numeric_imputer.fit_transform(clean_df[numeric_cols])
        
        # For categorical columns, use most frequent imputation
        cat_cols = clean_df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0] if not clean_df[col].mode().empty else "Unknown")
        
        # Log the remaining number of missing values
        remaining_missing = clean_df.isna().sum().sum()
        logger.info(f"Reduced missing values from {total_missing} to {remaining_missing}")
        
        return clean_df

    def save_processed_data(self, processed_data: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to CSV file.

        Args:
            processed_data: DataFrame with processed data
            filename: Name of the output file
        """
        try:
            # Check if processed_dir exists, if not create a default
            if self.processed_dir is None:
                self.processed_dir = os.path.join(self.data_dir, f'season_{self.season}', 'processed')
                logger.warning(f"processed_dir not set, defaulting to {self.processed_dir}")
                
            # Ensure directory exists
            os.makedirs(self.processed_dir, exist_ok=True)
            
            # Save the data
            file_path = os.path.join(self.processed_dir, filename)
            processed_data.to_csv(file_path, index=False)
            logger.info(f"Saved processed data to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            # Don't raise the exception to allow the pipeline to continue

    def process_qb_data(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process raw NFL data into QB analysis dataset.

        Args:
            raw_data: Dictionary containing raw NFL datasets

        Returns:
            Processed QB analysis dataset
        """
        try:
            pbp_data = raw_data.get('play_by_play', pd.DataFrame())
            player_stats = raw_data.get('player_stats', pd.DataFrame())
            player_info = raw_data.get('player_info', pd.DataFrame())
            team_info = raw_data.get('team_info', pd.DataFrame())
            weekly_stats = raw_data.get('weekly_stats', pd.DataFrame())
            ngs_data = raw_data.get('ngs_data', pd.DataFrame())
            
            # Validate we have the necessary data
            if player_stats.empty:
                raise ValueError("Player stats data is required for QB analysis")
            
            # Log data sizes
            logger.info(f"Processing QB data from: "
                        f"{len(pbp_data)} play-by-play records, "
                        f"{len(player_stats)} player stat records, "
                        f"{len(player_info)} player info records, "
                        f"{len(team_info)} team records")
            
            # Step 1: Ensure player stats has position information
            if 'position' not in player_stats.columns:
                logger.info("Merging player stats with player info to get position data")
                logger.info(f"Player stats columns: {player_stats.columns.tolist()[:10]}...")
                logger.info(f"Player info columns: {player_info.columns.tolist()[:10]}...")
                
                # Log some player IDs for debugging
                if 'player_id' in player_stats.columns:
                    logger.info(f"Player stats sample IDs: {player_stats['player_id'].head(3).tolist()}")
                
                # Try to create a common key for merging
                if 'player_id' in player_stats.columns and 'gsis_id' in player_info.columns:
                    logger.info("Using player_id to gsis_id mapping")
                    player_info['player_id'] = player_info['gsis_id']
                
                # Merge stats with player info
                player_stats = player_stats.merge(
                    player_info[['player_id', 'position']],
                    on='player_id',
                    how='left'
                )
                
                logger.info(f"After merge: {len(player_stats)} records, position nulls: {player_stats['position'].isna().sum()}")
            
            # Step 2: Filter to QBs with sufficient attempts
            qb_stats = self.filter_qb_data(player_stats)
            
            # Step 3: Prepare play-by-play data for QB analysis
            if not pbp_data.empty:
                pbp_qb_data = self.prepare_pbp_qb_data(pbp_data)
            else:
                pbp_qb_data = pd.DataFrame()
                logger.warning("No play-by-play data available for QB analysis")
            
            # Step 4: Enrich QB stats with play-by-play metrics
            enriched_stats = self.enrich_qb_stats(qb_stats, pbp_qb_data, player_info)
            
            # Step 5: Add team identifier if missing
            enriched_stats = self.add_team_identifier(enriched_stats, player_info)
            
            # Step 6: Add team context
            if team_info is not None and not team_info.empty:
                enriched_stats = self.add_team_context(enriched_stats, team_info)
            
            # Step 7: Handle missing values
            final_stats = self.handle_missing_values(enriched_stats)
            
            # Step 8: Save processed data
            self.save_processed_data(final_stats, 'qb_analysis_data.csv')
            
            if not pbp_qb_data.empty:
                self.save_processed_data(pbp_qb_data, 'prepared_pbp_data.csv')
            
            logger.info(f"Successfully processed QB data with {len(final_stats)} records")
            return final_stats
            
        except Exception as e:
            logger.error(f"Error processing QB data: {str(e)}")
            raise

    def add_team_identifier(self, qb_stats: pd.DataFrame, player_info: pd.DataFrame) -> pd.DataFrame:
        """
        Add team identifier to QB stats by matching with player info.
        
        Args:
            qb_stats: DataFrame with QB statistics
            player_info: DataFrame with player information
            
        Returns:
            DataFrame with team identifier added
        """
        # Create a copy to avoid modifying the original
        enhanced_stats = qb_stats.copy()
        
        # Check if we already have a team column
        if any(col in enhanced_stats.columns for col in ['recent_team', 'team', 'team_abbr']):
            logger.info("QB stats already contains a team identifier column")
            return enhanced_stats
            
        # Check if we have player_id column to join with player_info
        if 'player_id' not in enhanced_stats.columns:
            logger.warning("No player_id column in QB stats, cannot add team identifier")
            return enhanced_stats
            
        # Check if player_info has the needed columns
        if player_info is None or 'player_id' not in player_info.columns:
            logger.warning("Player info not available or missing player_id column")
            return enhanced_stats
            
        # Get team info from player_info
        if 'current_team_id' in player_info.columns:
            # Get mapping of team IDs to team abbreviations if available
            if 'team_id' in player_info.columns and 'team_abbr' in player_info.columns:
                team_mapping = player_info[['team_id', 'team_abbr']].drop_duplicates()
                
                # First get team_id, then map to abbreviation
                enhanced_stats = enhanced_stats.merge(
                    player_info[['player_id', 'current_team_id']],
                    on='player_id',
                    how='left'
                )
                
                # Rename for clarity
                enhanced_stats = enhanced_stats.rename(columns={'current_team_id': 'team_id'})
                
                # Map team_id to team_abbr
                enhanced_stats = enhanced_stats.merge(
                    team_mapping,
                    on='team_id',
                    how='left'
                )
                
                logger.info(f"Added team_abbr to {enhanced_stats['team_abbr'].notna().sum()} QB records")
                return enhanced_stats
            else:
                # Just add team_id if abbreviation mapping not available
                enhanced_stats = enhanced_stats.merge(
                    player_info[['player_id', 'current_team_id']],
                    on='player_id',
                    how='left'
                )
                
                # Rename for clarity
                enhanced_stats = enhanced_stats.rename(columns={'current_team_id': 'team_id'})
                logger.info(f"Added team_id to {enhanced_stats['team_id'].notna().sum()} QB records")
                return enhanced_stats
                
        logger.warning("No team information available in player_info")
        return enhanced_stats


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../data_collection')
    from data_fetcher import NFLDataFetcher
    
    # 1. Fetch the data
    fetcher = NFLDataFetcher(data_dir='../../data')
    raw_data = fetcher.get_qb_data(season=2023)
    
    # 2. Process the data
    processor = NFLDataProcessor(data_dir='../../data')
    processed_data = processor.process_qb_data(raw_data)
    
    # 3. Display summary
    print(f"Processed {len(processed_data)} QB records")
    print(f"Columns: {processed_data.columns.tolist()}") 