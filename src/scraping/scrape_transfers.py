import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from reus.fbref import fb_season_urls, fb_team_player_summary_stats

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_league_players(league_name):
    print(f"\n📌 Getting teams for: {league_name}")

    # 1. Get season overview URL
    season_urls = fb_season_urls(
        competition_name=league_name,
        competition_type="Domestic Leagues - 1st Tier",
        gender="M",
        season_end_year=2024
    )

    if season_urls.empty:
        print(f"❌ No season URL found for {league_name}")
        return pd.DataFrame()

    season_url = season_urls.iloc[0]
    print(f"  ✔️ Season URL: {season_url}")

    # 2. Parse season page to get team URLs
    r = requests.get(season_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    team_links = soup.select("table.stats_table a[href*='/squads/']")
    team_urls = ["https://fbref.com" + a["href"] for a in team_links]

    print(f"  ✔️ Found {len(team_urls)} teams.")

    if len(team_urls) == 0:
        print("❌ No team URLs extracted.")
        return pd.DataFrame()

    all_players = []

    # 3. Scrape each team page
    for team_url in team_urls:
        print(f"    🏟️ Scraping team: {team_url}")

        try:
            summary, _ = fb_team_player_summary_stats(url=team_url)
            df = pd.DataFrame(summary)

            if "player" in df.columns:
                df["team_url"] = team_url
                all_players.append(df[["player", "team_url"]])
        except Exception as e:
            print(f"      ❌ Team error: {e}")
            continue

        time.sleep(2)

    if not all_players:
        print("❌ No players collected.")
        return pd.DataFrame()

    players_df = pd.concat(all_players).drop_duplicates()
    print(f"\n✅ Successfully collected {len(players_df)} players from {league_name}")
    
    return players_df
