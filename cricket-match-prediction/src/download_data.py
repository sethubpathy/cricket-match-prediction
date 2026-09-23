"""Data acquisition script for downloading and compiling Cricsheet IPL match data."""

import io
import os
import urllib.request
import zipfile
import csv
import pandas as pd

CRICSHEET_IPL_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ZIP_DEST = os.path.join(RAW_DATA_DIR, "ipl_csv2.zip")
MATCHES_CSV = os.path.join(RAW_DATA_DIR, "matches.csv")
DATA_SOURCES_MD = os.path.join(os.path.dirname(__file__), "..", "data", "data_sources.md")


def download_and_extract():
    """Download Cricsheet IPL dataset and compile matches.csv."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print(f"Downloading IPL dataset from {CRICSHEET_IPL_URL}...")
    req = urllib.request.Request(CRICSHEET_IPL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()

    with open(ZIP_DEST, "wb") as f:
        f.write(content)
    print(f"Saved raw archive to {ZIP_DEST} ({len(content) / (1024 * 1024):.2f} MB)")

    print("Parsing match info metadata into matches.csv...")
    matches = []

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        info_files = [n for n in z.namelist() if n.endswith("_info.csv")]
        print(f"Found {len(info_files)} match metadata files.")

        for fname in info_files:
            file_bytes = z.read(fname).decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(file_bytes))

            match_data = {
                "match_id": None,
                "season": None,
                "date": None,
                "city": None,
                "venue": None,
                "team1": None,
                "team2": None,
                "toss_winner": None,
                "toss_decision": None,
                "winner": None,
                "eliminator": None,
                "result": "normal",
                "win_by_runs": 0,
                "win_by_wickets": 0,
                "method": None,
                "player_of_match": None,
            }

            teams = []
            dates = []
            player_of_match_list = []

            for row in reader:
                if not row or row[0] != "info":
                    continue
                key = row[1]
                val = row[2] if len(row) > 2 else None

                if key == "match_id":
                    match_data["match_id"] = val
                elif key == "season":
                    match_data["season"] = val
                elif key == "date":
                    dates.append(val)
                elif key == "city":
                    match_data["city"] = val
                elif key == "venue":
                    match_data["venue"] = val
                elif key == "team":
                    if val not in teams:
                        teams.append(val)
                elif key == "toss_winner":
                    match_data["toss_winner"] = val
                elif key == "toss_decision":
                    match_data["toss_decision"] = val
                elif key == "winner":
                    match_data["winner"] = val
                elif key == "winner_runs":
                    match_data["win_by_runs"] = int(val) if val and val.isdigit() else 0
                elif key == "winner_wickets":
                    match_data["win_by_wickets"] = int(val) if val and val.isdigit() else 0
                elif key == "outcome":
                    match_data["result"] = val
                elif key == "eliminator":
                    match_data["eliminator"] = val
                    if not match_data["winner"]:
                        match_data["winner"] = val
                elif key == "method":
                    match_data["method"] = val
                elif key == "player_of_match":
                    player_of_match_list.append(val)

            # Fallback match id from filename if missing
            if not match_data["match_id"]:
                match_data["match_id"] = os.path.basename(fname).replace("_info.csv", "")

            if dates:
                match_data["date"] = dates[0]
            if len(teams) >= 2:
                match_data["team1"] = teams[0]
                match_data["team2"] = teams[1]
            elif len(teams) == 1:
                match_data["team1"] = teams[0]

            if player_of_match_list:
                match_data["player_of_match"] = player_of_match_list[0]

            matches.append(match_data)

    df = pd.DataFrame(matches)
    # Sort chronologically
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by=["date_dt", "match_id"]).drop(columns=["date_dt"]).reset_index(drop=True)

    df.to_csv(MATCHES_CSV, index=False)
    print(f"Saved compiled matches to {MATCHES_CSV}")

    # Write data_sources.md
    min_date = df["date"].dropna().min()
    max_date = df["date"].dropna().max()
    seasons = sorted([str(s) for s in df["season"].dropna().unique()])

    data_sources_content = f"""# Data Sources Documentation

## Primary Dataset: Cricsheet Indian Premier League (IPL) Matches
- **Source**: [Cricsheet.org](https://cricsheet.org/downloads/)
- **Download URL**: `{CRICSHEET_IPL_URL}`
- **License**: [Open Data Commons Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/)
- **Match Format**: Twenty20 (T20) — Indian Premier League
- **Date Range Covered**: {min_date} to {max_date} (Seasons: {seasons[0]} – {seasons[-1]})
- **Total Records**: {len(df)} matches
- **Total Columns**: {len(df.columns)}

## Raw Schema (`data/raw/matches.csv`)
| Column | Description |
| :--- | :--- |
| `match_id` | Unique numerical identifier for each match from Cricsheet |
| `season` | IPL tournament edition/year |
| `date` | Date of the match (YYYY/MM/DD) |
| `city` | Host city |
| `venue` | Stadium/ground name |
| `team1` | First team (as recorded in match info) |
| `team2` | Second team |
| `toss_winner` | Team winning the coin toss |
| `toss_decision` | Decision taken upon winning toss (`bat` or `field`) |
| `winner` | Match winner |
| `eliminator` | Winner determined by Super Over / Eliminator (if tie) |
| `result` | Result descriptor (`normal`, `tie`, `no result`) |
| `win_by_runs` | Margin of victory in runs (if batting first) |
| `win_by_wickets` | Margin of victory in wickets (if chasing) |
| `method` | DLS / Rain adjustment method (if applicable) |
| `player_of_match` | Recipient of Player of the Match |
"""

    with open(DATA_SOURCES_MD, "w", encoding="utf-8") as f:
        f.write(data_sources_content)
    print(f"Created documentation at {DATA_SOURCES_MD}")

    # Print summary output requested by Step 2 prompt
    print("\n" + "=" * 50)
    print("STEP 2 OUTPUT: RAW DATA SHAPE & COLUMNS")
    print("=" * 50)
    print(f"Dataset Shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print("\nColumn List:")
    for idx, col in enumerate(df.columns, 1):
        sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
        print(f"  {idx:2d}. {col:<18} (e.g. {sample_val})")
    print("=" * 50)

    print("\nHead (First 3 rows):")
    print(df[["match_id", "season", "date", "team1", "team2", "toss_winner", "toss_decision", "winner"]].head(3).to_string())


if __name__ == "__main__":
    download_and_extract()
