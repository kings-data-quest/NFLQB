#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL QB Data Fetcher Module

This module handles the collection of NFL quarterback data using nfl-data-py,
a Python wrapper for NFLfastR.
"""

import os
import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import nfl_data_py as nfl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLDataFetcher:
    """Class for fetching NFL data using nfl-data-py package."""

    def __init__(self, data_dir: str = '../data'):
        """
        Initialize the NFL data fetcher.

        Args:
            data_dir: Directory to store the downloaded data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Initialized NFL data fetcher with data directory: {data_dir}")

    def get_play_by_play_data(self, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch play-by-play data for specified seasons.

        Args:
            seasons: List of seasons (years) to fetch data for

        Returns:
            DataFrame containing play-by-play data
        """
        logger.info(f"Fetching play-by-play data for seasons: {seasons}")
        try:
            pbp_data = nfl.import_pbp_data(seasons)
            logger.info(f"Successfully fetched {len(pbp_data)} play-by-play records")
            return pbp_data
        except Exception as e:
            logger.error(f"Error fetching play-by-play data: {str(e)}")
            raise

    def get_player_stats(self, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch player stats for specified seasons.

        Args:
            seasons: List of seasons (years) to fetch data for

        Returns:
            DataFrame containing player stats
        """
        logger.info(f"Fetching player stats for seasons: {seasons}")
        try:
            player_stats = nfl.import_seasonal_data(seasons)
            logger.info(f"Successfully fetched stats for {len(player_stats)} player records")
            return player_stats
        except Exception as e:
            logger.error(f"Error fetching player stats: {str(e)}")
            raise

    def get_next_gen_stats(self, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch NFL Next Gen Stats for specified seasons.

        Args:
            seasons: List of seasons (years) to fetch data for

        Returns:
            DataFrame containing Next Gen Stats data
        """
        logger.info(f"Fetching Next Gen Stats for seasons: {seasons}")
        try:
            # Update to specifically request 'passing' stats for QBs
            ngs_data = nfl.import_ngs_data(seasons, stat_type='passing')
            logger.info(f"Successfully fetched {len(ngs_data)} Next Gen Stats records")
            return ngs_data
        except Exception as e:
            logger.error(f"Error fetching Next Gen Stats: {str(e)}")
            # Instead of raising, return an empty DataFrame with a warning
            logger.warning(f"Couldn't fetch NGS data: {str(e)}, continuing without it")
            return pd.DataFrame()

    def get_player_info(self) -> pd.DataFrame:
        """
        Fetch player information (metadata).

        Returns:
            DataFrame containing player information
        """
        logger.info("Fetching player information")
        try:
            player_info = nfl.import_players()
            logger.info(f"Successfully fetched info for {len(player_info)} players")
            return player_info
        except Exception as e:
            logger.error(f"Error fetching player information: {str(e)}")
            raise

    def get_team_info(self, seasons=None) -> pd.DataFrame:
        """
        Fetch team information, including win-loss records if seasons are provided.

        Args:
            seasons: Optional list of seasons to fetch record information for

        Returns:
            DataFrame containing team information with record data
        """
        logger.info("Fetching team information")
        try:
            # Get basic team info
            team_info = nfl.import_team_desc()
            logger.info(f"Successfully fetched info for {len(team_info)} teams")
            
            # If seasons are provided, get win-loss records
            if seasons and isinstance(seasons, list):
                logger.info(f"Fetching team records for seasons: {seasons}")
                try:
                    # Get schedule data to calculate records
                    schedules = nfl.import_schedules(seasons)
                    
                    # Only use regular season games (game_type == 'REG')
                    reg_season = schedules[schedules['game_type'] == 'REG'].copy()
                    
                    # Initialize records dictionary
                    team_records = {}
                    
                    # Process schedule to calculate team records
                    for _, game in reg_season.iterrows():
                        home_team = game['home_team']
                        away_team = game['away_team']
                        
                        # Skip games with missing scores or unplayed games
                        if pd.isna(game['home_score']) or pd.isna(game['away_score']):
                            continue
                            
                        home_score = int(game['home_score'])
                        away_score = int(game['away_score'])
                        
                        # Initialize team records if not already present
                        for team in [home_team, away_team]:
                            if team not in team_records:
                                team_records[team] = {'wins': 0, 'losses': 0, 'ties': 0, 'points_for': 0, 'points_against': 0}
                        
                        # Home team result
                        if home_score > away_score:
                            team_records[home_team]['wins'] += 1
                            team_records[away_team]['losses'] += 1
                        elif home_score < away_score:
                            team_records[home_team]['losses'] += 1
                            team_records[away_team]['wins'] += 1
                        else:
                            team_records[home_team]['ties'] += 1
                            team_records[away_team]['ties'] += 1
                        
                        # Update points
                        team_records[home_team]['points_for'] += home_score
                        team_records[home_team]['points_against'] += away_score
                        team_records[away_team]['points_for'] += away_score
                        team_records[away_team]['points_against'] += home_score
                    
                    # Convert records to DataFrame
                    records_df = pd.DataFrame.from_dict(team_records, orient='index').reset_index()
                    records_df.rename(columns={'index': 'team_abbr'}, inplace=True)
                    
                    # Merge with team_info
                    team_info = team_info.merge(records_df, on='team_abbr', how='left')
                    
                    # Fill NAs for teams with no games
                    for col in ['wins', 'losses', 'ties', 'points_for', 'points_against']:
                        team_info[col] = team_info[col].fillna(0)
                    
                    logger.info(f"Added team records from {len(seasons)} seasons")
                
                except Exception as e:
                    logger.warning(f"Could not fetch team records: {str(e)}. Continuing with basic team info.")
            
            return team_info
        except Exception as e:
            logger.error(f"Error fetching team information: {str(e)}")
            raise
            
    def get_weekly_stats(self, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch weekly player stats for specified seasons.

        Args:
            seasons: List of seasons (years) to fetch data for

        Returns:
            DataFrame containing weekly player stats
        """
        logger.info(f"Fetching weekly player stats for seasons: {seasons}")
        try:
            weekly_data = nfl.import_weekly_data(seasons)
            logger.info(f"Successfully fetched {len(weekly_data)} weekly stat records")
            return weekly_data
        except Exception as e:
            logger.error(f"Error fetching weekly stats: {str(e)}")
            raise

    def get_qb_data(self, season: int = 2023) -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive QB data for the specified season.
        
        This is a convenience method that fetches all relevant data
        for QB analysis and returns it as a dictionary of DataFrames.

        Args:
            season: NFL season year (default: 2023)

        Returns:
            Dictionary containing different datasets relevant for QB analysis
        """
        logger.info(f"Fetching comprehensive QB data for season {season}")
        try:
            seasons = [season]
            
            # Fetch all relevant datasets
            play_by_play = self.get_play_by_play_data(seasons)
            player_stats = self.get_player_stats(seasons)
            player_info = self.get_player_info()
            team_info = self.get_team_info(seasons)
            weekly_stats = self.get_weekly_stats(seasons)
            
            # Try to get NGS data if available (not always complete for recent seasons)
            try:
                ngs_data = self.get_next_gen_stats(seasons)
            except Exception as e:
                logger.warning(f"Couldn't fetch NGS data: {str(e)}, continuing without it")
                ngs_data = pd.DataFrame()
            
            # Save raw data files to disk
            self._save_dataframes({
                'play_by_play': play_by_play,
                'player_stats': player_stats,
                'player_info': player_info,
                'team_info': team_info,
                'weekly_stats': weekly_stats,
                'ngs_data': ngs_data
            }, season)
            
            return {
                'play_by_play': play_by_play,
                'player_stats': player_stats,
                'player_info': player_info,
                'team_info': team_info,
                'weekly_stats': weekly_stats,
                'ngs_data': ngs_data
            }
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive QB data: {str(e)}")
            raise

    def _save_dataframes(self, dataframes: Dict[str, pd.DataFrame], season: int) -> None:
        """
        Save each DataFrame to a CSV file.

        Args:
            dataframes: Dictionary mapping dataset names to DataFrames
            season: Season year for filename
        """
        season_dir = os.path.join(self.data_dir, f'season_{season}', 'raw')
        os.makedirs(season_dir, exist_ok=True)
        
        for name, df in dataframes.items():
            if not df.empty:
                file_path = os.path.join(season_dir, f'{name}.csv')
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {name} data to {file_path}")


if __name__ == "__main__":
    # Example usage
    fetcher = NFLDataFetcher(data_dir='../../data')
    
    # Fetch data for 2023 season
    season_data = fetcher.get_qb_data(season=2023)
    
    # Display summary information
    for name, df in season_data.items():
        if not df.empty:
            print(f"{name}: {len(df)} records, {df.shape[1]} columns") 