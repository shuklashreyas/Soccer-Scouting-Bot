# Soccer-Scouting-Bot

**Purpose:** A small toolkit and Streamlit app for exploring player profiles, clustering player roles, and finding statistically similar players using per-player features derived from football statistics.

**Quick Setup**
- **Create & activate virtualenv:**
	- `python -m venv .venv`
	- `source .venv/bin/activate` (zsh)
- **Install dependencies:**
	- `pip install -r requirements.txt`

**Run the app (development)**
- From project root run:
	- `streamlit run src/app/app.py`

**Scraping / Data**
- Scrapers are in `src/scraping/`.
- Example (from project root):
	- `python src/scraping/scrape_fbref.py`
	- `python src/scraping/scrape_transfers.py`
- Scraped and processed outputs are placed under `data/` and `data/models/`.

**Training / Models**
- Quick training helpers:
	- `python -m src.modeling.train_player_embedding` — builds and saves the `PlayerEmbeddingModel` to `data/models`.
	- `python src/modeling/train_similarity_model.py` — creates a simple kNN similarity model and saves to `data/models`.

**Tests**
- Run unit tests:
	- `pytest -q`

**Project structure (high-level)**
- `src/app/` — Streamlit app and admin tools
- `src/scraping/` — scrapers for FBref and transfers
- `src/preprocessing/` — data cleaning scripts
- `src/modeling/` — clustering, similarity, and training scripts
- `src/nlp/` — intent/entity helpers for the UI
- `src/player/` — player extraction, lookup, insights
- `data/` — raw and processed CSVs and saved models

**Notes & Recommendations**
- I generated a small, curated `requirements-min.txt` with packages actually imported by the code. You can use that to install a minimal environment for running the app and tests. If you prefer pinned versions, I can replace `requirements.txt` with a pinned minimal list copied from the original file.
- Some modules (training scripts, scraping with Selenium, or heavy ML frameworks) may require additional system dependencies (ChromeDriver for Selenium, compiled SciPy/NumPy wheels, etc.).

# Deployment 
- https://shuklashreyas-soccer-scouting-bot-srcappapp-jewwpm.streamlit.app/
