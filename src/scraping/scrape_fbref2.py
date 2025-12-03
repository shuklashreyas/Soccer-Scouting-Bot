import pandas as pd
from io import StringIO
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
pd.set_option('display.max_columns', None)

def clean_column_names(df):
    """
    Clean column names to match FBref website display names.
    Handles multi-level columns and removes prefixes/suffixes.

    Parameters:
    - df: DataFrame with potentially messy column names

    Returns:
    - DataFrame with cleaned column names
    """
    if isinstance(df.columns, pd.MultiIndex):
        # For multi-level columns, join with underscore and clean
        new_cols = []
        for col in df.columns:
            # Join the multi-level column parts
            col_parts = [str(c) for c in col if str(c) not in ['Unnamed: 0_level_0', 'Unnamed: 1_level_0', 'Unnamed']]
            col_parts = [c for c in col_parts if not c.startswith('Unnamed')]

            # Join with underscore
            if col_parts:
                clean_name = '_'.join(col_parts)
            else:
                clean_name = '_'.join([str(c) for c in col])

            new_cols.append(clean_name)

        df.columns = new_cols

    # Clean up column names
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)

        # Remove common prefixes that pandas adds
        col_str = col_str.replace('Unnamed: ', '')
        col_str = col_str.replace('_level_0', '')
        col_str = col_str.replace('_level_1', '')

        # Remove leading/trailing underscores and whitespace
        col_str = col_str.strip('_').strip()

        # Replace multiple underscores with single
        while '__' in col_str:
            col_str = col_str.replace('__', '_')

        cleaned_columns.append(col_str)

    df.columns = cleaned_columns

    return df


def get_fbref_table_selenium(url, wait_time=10):
    """
    Fetch individual player data table from FBref using Selenium.
    Gets the SECOND table (player stats), not the first (league comparison).

    Parameters:
    - url: Full URL of the page
    - wait_time: Time to wait for page to load

    Returns:
    - DataFrame of individual player stats with cleaned column names
    """

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)

        # Wait for table to load
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )

        # Additional wait to ensure JavaScript loads all player data
        time.sleep(5)

        # Get page source and parse ALL tables
        html = driver.page_source
        dfs = pd.read_html(StringIO(html))

        # Get the SECOND table (index 1) which contains player statistics
        # First table (index 0) is the league comparison table
        if len(dfs) < 2:
            player_df = dfs[0]
        else:
            player_df = dfs[1]

        # Clean column names to match FBref display
        player_df = clean_column_names(player_df)

        # Remove any rows that are just headers repeated in the middle of the table
        if 'Player' in player_df.columns:
            player_df = player_df[player_df['Player'] != 'Player']

        return player_df

    except Exception as e:
        print(f"  Error: {e}")
        return None
    finally:
        if driver:
            driver.quit()


def scrape_big5_individual_players(season="2024-2025"):
    """
    Scrape individual player statistics (NOT league averages) from FBref.

    Parameters:
    - season: Season string (e.g., "2024-2025")

    Returns:
    - Dictionary of DataFrames with individual player stats for each category
    """

    # URLs for INDIVIDUAL PLAYER DATA for specific season
    stat_categories = {
        'standard': f'https://fbref.com/en/comps/Big5/{season}/stats/players/{season}-Big-5-European-Leagues-Stats',
        'shooting': f'https://fbref.com/en/comps/Big5/{season}/shooting/players/{season}-Big-5-European-Leagues-Stats',
        'passing': f'https://fbref.com/en/comps/Big5/{season}/passing/players/{season}-Big-5-European-Leagues-Stats',
        'passing_types': f'https://fbref.com/en/comps/Big5/{season}/passing_types/players/{season}-Big-5-European-Leagues-Stats',
        'gca': f'https://fbref.com/en/comps/Big5/{season}/gca/players/{season}-Big-5-European-Leagues-Stats',
        'defense': f'https://fbref.com/en/comps/Big5/{season}/defense/players/{season}-Big-5-European-Leagues-Stats',
        'possession': f'https://fbref.com/en/comps/Big5/{season}/possession/players/{season}-Big-5-European-Leagues-Stats',
        'keepers': f'https://fbref.com/en/comps/Big5/{season}/keepers/players/{season}-Big-5-European-Leagues-Stats',
        'keepersadv': f'https://fbref.com/en/comps/Big5/{season}/keepersadv/players/{season}-Big-5-European-Leagues-Stats',
    }

    all_data = {}

    for i, (category, url) in enumerate(stat_categories.items(), 1):

        df = get_fbref_table_selenium(url)

        if df is not None and not df.empty:
            all_data[category] = df

            # Show sample to verify we got individual players
            if 'Player' in df.columns:
                sample_players = df['Player'].head(3).tolist()


    return all_data


def merge_outfield_stats(all_data):
    """
    Merge all OUTFIELD player stat categories (excluding goalkeeper stats).

    Parameters:
    - all_data: Dictionary of DataFrames from scraper

    Returns:
    - Single merged DataFrame with all outfield player stats
    """
    # Categories to include in outfield player stats (exclude keeper stats)
    outfield_categories = ['standard', 'shooting', 'passing', 'passing_types',
                           'gca', 'defense', 'possession', 'playingtime', 'misc']

    # Filter to only outfield categories that exist
    outfield_data = {k: v for k, v in all_data.items() if k in outfield_categories}

    if not outfield_data:
        return None

    # Start with standard stats as base
    if 'standard' not in outfield_data:
        base_df = list(outfield_data.values())[0].copy()
    else:
        base_df = outfield_data['standard'].copy()

    # Key columns for merging
    possible_keys = ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born']
    merge_keys = [k for k in possible_keys if k in base_df.columns]

    if not merge_keys:
        return outfield_data

    for category, df in outfield_data.items():
        if category == 'standard':
            continue

        common_keys = [k for k in merge_keys if k in df.columns]

        if common_keys:
            before_cols = len(base_df.columns)
            base_df = base_df.merge(
                df,
                on=common_keys,
                how='left',
                suffixes=('', f'_{category}')
            )
            after_cols = len(base_df.columns)
            added_cols = after_cols - before_cols

    return base_df


def merge_keeper_stats(all_data):
    """
    Merge GOALKEEPER stat categories only.

    Parameters:
    - all_data: Dictionary of DataFrames from scraper

    Returns:
    - Single merged DataFrame with all goalkeeper stats
    """
    # Get only keeper categories
    keeper_categories = ['keepers', 'keepersadv']
    keeper_data = {k: v for k, v in all_data.items() if k in keeper_categories}

    if not keeper_data:
        return None

    # Start with basic keeper stats if available
    if 'keepers' in keeper_data:
        base_df = keeper_data['keepers'].copy()
        start_category = 'keepers'
    else:
        base_df = keeper_data['keepersadv'].copy()
        start_category = 'keepersadv'

    # Key columns for merging
    possible_keys = ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born']
    merge_keys = [k for k in possible_keys if k in base_df.columns]

    if not merge_keys:
        return keeper_data

    for category, df in keeper_data.items():
        if category == start_category:
            continue

        common_keys = [k for k in merge_keys if k in df.columns]

        if common_keys:
            before_cols = len(base_df.columns)
            base_df = base_df.merge(
                df,
                on=common_keys,
                how='left',
                suffixes=('', f'_{category}')
            )
            after_cols = len(base_df.columns)
            added_cols = after_cols - before_cols

    return base_df

# Split Outfield and Goalkeeper Data
player_data = scrape_big5_individual_players(season="2024-2025")
outfield_df = merge_outfield_stats(player_data)
keeper_df = merge_keeper_stats(player_data)

# Filter for needed columns
outfield_df.fillna(value=0, inplace=True)
selected_columns = [column for column in outfield_df.columns if not ((column.startswith('Performance')) or (column.startswith('Matches')) or (column.startswith('Rk')) or ('90s' in column) or ('Per 90' in column) or ('90' in column))]
selected_columns.append('90s')

outfield2 = outfield_df[selected_columns]


# Change from statistics from total to per 90
per_90_cols = [col for col in outfield2.columns[10:-1] if not ('%' in col)]
outfield2[per_90_cols] = outfield2[per_90_cols].astype(float)
outfield2['90s'] = outfield2['90s'].astype(float)
for col in per_90_cols:
    outfield2[col] = outfield2[col] / outfield2['90s']

# Clean Outfield Data
prefixes_to_remove = ['Standard_', 'Expected_', 'Playing Time_', 'Progression_', 'Total_']
for prefix in prefixes_to_remove:
    outfield2.columns = outfield2.columns.str.replace(prefix, '')
outfield2.sort_values(by='90s', ascending=False, inplace=True)
outfield2 = outfield2[outfield2['Pos'] != 'GK']

# Drop players with missing data
outfield2.dropna(axis=0, inplace=True)
outfield2 = outfield2.round(3)

# Save CSV
outfield2.to_csv('../../data/processed/outfield_clean.csv')