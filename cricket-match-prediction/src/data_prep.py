"""Data preparation and cleaning module for Cricket Match Prediction."""

import os
import pandas as pd

# Canonical franchise name standardization mapping
TEAM_MAPPINGS = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Venue name harmonization mapping to prevent stadium fragmentation
VENUE_MAPPINGS = {
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "Shaheed Veer Narayan Singh International Stadium, Raipur": "Shaheed Veer Narayan Singh International Stadium",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium",
}

# Mapping to impute missing host cities based on known stadium locations
VENUE_TO_CITY = {
    "Dubai International Cricket Stadium": "Dubai",
    "Sharjah Cricket Stadium": "Sharjah",
    "Sheikh Zayed Stadium": "Abu Dhabi",
    "Zayed Cricket Stadium, Abu Dhabi": "Abu Dhabi",
}


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize historical cricket match data.

    Data Cleaning Choices & Documentation:
    1. Standardize Team Names:
       - Renamed franchises (Delhi Daredevils -> Delhi Capitals, Kings XI Punjab -> Punjab Kings,
         Royal Challengers Bangalore -> Royal Challengers Bengaluru, Rising Pune Supergiants -> Rising Pune Supergiant)
         are standardized across team1, team2, toss_winner, winner, and eliminator so historical
         records and head-to-head stats remain continuous.
    2. Standardize Venue Names:
       - Ground name variations and corporate sponsor changes (e.g. Feroz Shah Kotla -> Arun Jaitley Stadium,
         stadiums with trailing city names) are mapped to consistent canonical names.
    3. Remove Abandoned / No-Result Matches:
       - Matches with result == 'no result' or null winner (9 matches) are dropped because they offer zero
         target signal for predictive outcome modeling.
    4. Handle Missing Values:
       - 'city': 51 missing values at UAE stadiums are imputed using stadium-to-city lookup.
       - 'eliminator': Filled with 'None' for matches decided in normal regulation overs.
       - 'method': Filled with 'Normal' for non-DLS rain-shortened matches.
       - 'player_of_match': Filled with 'None' if unrecorded.
    5. Datetime Conversion & Chronological Sorting:
       - Date strings are parsed to pd.Timestamp and rows are sorted chronologically by date and match_id.
    """
    cleaned = df.copy()

    # 1. Standardize team names across all team-related columns
    team_cols = ["team1", "team2", "toss_winner", "winner", "eliminator"]
    for col in team_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].replace(TEAM_MAPPINGS)

    # 2. Standardize venue names
    if "venue" in cleaned.columns:
        cleaned["venue"] = cleaned["venue"].replace(VENUE_MAPPINGS)

    # 3. Filter out matches with no result (rain-outs, abandoned matches)
    cleaned = cleaned[cleaned["winner"].notnull() & (cleaned["result"] != "no result")].copy()

    # 4. Handle missing values
    if "city" in cleaned.columns:
        cleaned["city"] = cleaned["city"].fillna(cleaned["venue"].map(VENUE_TO_CITY))

    if "eliminator" in cleaned.columns:
        cleaned["eliminator"] = cleaned["eliminator"].fillna("None")

    if "method" in cleaned.columns:
        cleaned["method"] = cleaned["method"].fillna("Normal")

    if "player_of_match" in cleaned.columns:
        cleaned["player_of_match"] = cleaned["player_of_match"].fillna("None")

    # 5. Convert date to datetime and sort chronologically
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    cleaned = cleaned.sort_values(by=["date", "match_id"]).reset_index(drop=True)

    return cleaned


def run_data_prep():
    """Load raw data, run clean_matches, save cleaned data, and print before/after stats."""
    candidate_paths = [
        os.path.abspath("data/raw/matches.csv"),
        os.path.abspath("cricket-match-prediction/data/raw/matches.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "matches.csv"),
    ]

    raw_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            raw_path = p
            break

    if not raw_path:
        raise FileNotFoundError(f"Cannot find raw matches.csv in: {candidate_paths}")

    print(f"Loading raw match dataset from: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    before_rows = len(raw_df)

    print("Cleaning matches dataset...")
    clean_df = clean_matches(raw_df)
    after_rows = len(clean_df)

    # Save to data/processed/matches_clean.csv (in both target folders for safety)
    output_locations = [
        os.path.abspath("data/processed/matches_clean.csv"),
        os.path.abspath("cricket-match-prediction/data/processed/matches_clean.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "matches_clean.csv"),
    ]

    saved_paths = set()
    for out_path in output_locations:
        norm_path = os.path.normpath(out_path)
        if norm_path not in saved_paths:
            os.makedirs(os.path.dirname(norm_path), exist_ok=True)
            clean_df.to_csv(norm_path, index=False)
            saved_paths.add(norm_path)
            print(f"Saved cleaned data to: {norm_path}")

    print("\n" + "=" * 55)
    print("STEP 4 DATA CLEANING SUMMARY")
    print("=" * 55)
    print(f"Raw Row Count (Before):     {before_rows}")
    print(f"Cleaned Row Count (After):   {after_rows}")
    print(f"Rows Dropped (No Result):    {before_rows - after_rows}")
    print(f"Remaining Missing Values:    {clean_df.isnull().sum().sum()}")
    print(f"Cleaned Columns:             {list(clean_df.columns)}")
    print(f"Date Range:                  {clean_df['date'].min().strftime('%Y-%m-%d')} to {clean_df['date'].max().strftime('%Y-%m-%d')}")
    print("=" * 55)

    return clean_df


if __name__ == "__main__":
    run_data_prep()
