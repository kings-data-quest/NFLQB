import pandas as pd
import os

print("Checking NFL data files...\n")

# Load player stats
print("PLAYER STATS DATA:")
stats_df = pd.read_csv('../data/season_2023/raw/player_stats.csv')
print(f"Shape: {stats_df.shape}")
print(f"Columns: {stats_df.columns.tolist()}\n")
print("First 2 rows of player_stats:")
print(stats_df.head(2).to_string())
print("\n" + "-"*80 + "\n")

# Load player info
print("PLAYER INFO DATA:")
info_df = pd.read_csv('../data/season_2023/raw/player_info.csv')
print(f"Shape: {info_df.shape}")
print(f"Columns: {info_df.columns.tolist()}\n")
print("First 2 rows of player_info:")
print(info_df.head(2).to_string())
print("\n" + "-"*80 + "\n")

# Check if position column exists in either dataframe
print("Position column check:")
if 'position' in stats_df.columns:
    print("- 'position' column exists in player_stats")
else:
    print("- 'position' column does NOT exist in player_stats")

if 'position' in info_df.columns:
    print("- 'position' column exists in player_info")
else:
    print("- 'position' column does NOT exist in player_info")

# Check for other potential position-related columns
print("\nPotential position-related columns:")
for col in stats_df.columns:
    if 'pos' in col.lower():
        print(f"- In player_stats: '{col}'")
for col in info_df.columns:
    if 'pos' in col.lower():
        print(f"- In player_info: '{col}'") 