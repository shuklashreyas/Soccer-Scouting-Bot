import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
from io import StringIO
import os
import re

def scrape_fbref(league_url, league_name):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://fbref.com/",
        "Connection": "keep-alive",
    }

    response = requests.get(league_url, headers=headers, timeout=20)
    soup = BeautifulSoup(response.text, "html.parser")

    # FBref hides tables in HTML comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    df = None
    for comment in comments:
        if re.search(r'id="stats_standard[^"]*"', comment) or re.search(r'id="stats_standard_dom_lg[^"]*"', comment):
            try:
                table_soup = BeautifulSoup(comment, "html.parser")
                table = table_soup.find("table", id=re.compile(r"^(stats_standard|stats_standard_dom_lg)"))
                if table:
                    df = pd.read_html(StringIO(str(table)))[0]
                    break
            except Exception as e:
                print(f"⚠️ Error parsing table for {league_name}: {e}")

    if df is None:
        raise ValueError(f"No table found for {league_name}")

    # Flatten multi-level columns
    df.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df.columns
    ]

    # Find the Player column dynamically
    player_col = next((c for c in df.columns if "Player" in c), None)
    if not player_col:
        raise ValueError(f"'Player' column not found for {league_name}")

    # Clean
    df = df[df[player_col] != "Player"]
    df.reset_index(drop=True, inplace=True)
    df["League"] = league_name

    print(f"✅ Scraped {len(df)} players from {league_name}")
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

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
        output_path = "data/all_leagues_stats.csv"
        full_df.to_csv(output_path, index=False)
        print(f"💾 Saved all league data to {output_path}")
    else:
        print("❌ No league data scraped.")
