import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
from io import StringIO
import os

def scrape_fbref(league_url, league_name):
    response = requests.get(league_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # FBref hides the table inside comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    df = None

    for comment in comments:
        if 'table' in comment and 'stats_standard' in comment:
            table_soup = BeautifulSoup(comment, 'html.parser')
            table = table_soup.find('table')
            if table:
                df = pd.read_html(StringIO(str(table)))[0]
                break

    if df is None:
        raise ValueError(f"No table found for {league_name}")

    # Clean up
    df = df[df['Player'] != 'Player']
    df.reset_index(drop=True, inplace=True)
    df['League'] = league_name
    print(f"✅ Scraped {len(df)} players from {league_name}")
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    leagues = {
        "Premier League": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
        "La Liga": "https://fbref.com/en/comps/12/stats/La-Liga-Stats",
        "Bundesliga": "https://fbref.com/en/comps/20/stats/Bundesliga-Stats",
        "Serie A": "https://fbref.com/en/comps/11/stats/Serie-A-Stats",
        "Ligue 1": "https://fbref.com/en/comps/13/stats/Ligue-1-Stats",
    }

    all_data = []
    for league, url in leagues.items():
        try:
            df = scrape_fbref(url, league)
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Failed to scrape {league}: {e}")

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_csv("data/raw/all_leagues_stats.csv", index=False)
        print("💾 Saved all league data to data/raw/all_leagues_stats.csv")
