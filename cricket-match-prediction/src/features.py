"""Feature engineering module for Cricket Match Prediction."""

import os
from collections import defaultdict
import pandas as pd

# Primary home grounds for IPL franchises to compute home/away advantage
FRANCHISE_HOME_VENUES = {
    "Chennai Super Kings": ["MA Chidambaram Stadium"],
    "Mumbai Indians": ["Wankhede Stadium", "Brabourne Stadium"],
    "Royal Challengers Bengaluru": ["M Chinnaswamy Stadium"],
    "Kolkata Knight Riders": ["Eden Gardens"],
    "Delhi Capitals": ["Arun Jaitley Stadium"],
    "Punjab Kings": [
        "Punjab Cricket Association IS Bindra Stadium",
        "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    ],
    "Rajasthan Royals": ["Sawai Mansingh Stadium"],
    "Sunrisers Hyderabad": ["Rajiv Gandhi International Stadium"],
    "Gujarat Titans": ["Narendra Modi Stadium, Ahmedabad"],
    "Lucknow Super Giants": [
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow"
    ],
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictive features from cleaned match records without data leakage.

    Strict Leakage Prevention:
    Historical features for each match at index i are strictly computed from matches
    played BEFORE index i (prior to match date). State updates occur only AFTER feature
    generation for that match.

    Engineered Features:
    1. Rolling Form:
       - team1_form_5, team2_form_5: Win-rate over previous 5 matches
       - team1_form_10, team2_form_10: Win-rate over previous 10 matches
       - form_diff_5: (team1_form_5 - team2_form_5)
    2. Overall Historical Form:
       - team1_overall_win_rate, team2_overall_win_rate
       - overall_win_rate_diff: (team1_overall_win_rate - team2_overall_win_rate)
    3. Head-to-Head (H2H):
       - h2h_matches: Prior count of matches between team1 and team2
       - team1_h2h_win_rate: Proportion of prior H2H matches won by team1
    4. Venue Win-Rate:
       - team1_venue_win_rate, team2_venue_win_rate: Historical win rate of each team at this ground
       - venue_win_rate_diff: (team1_venue_win_rate - team2_venue_win_rate)
    5. Toss Dynamics:
       - team1_won_toss: 1 if team1 won the coin toss, 0 otherwise
       - toss_decision: 'bat' or 'field'
       - team1_batting_first: 1 if team1 is batting in the first innings, 0 if chasing
    6. Home / Away Indicator:
       - team1_is_home, team2_is_home: Binary indicator of playing at designated home stadium
    7. Target Variable:
       - team1_won: 1 if team1 won the match, 0 if team2 won
    """
    sorted_df = df.copy()
    sorted_df["date"] = pd.to_datetime(sorted_df["date"])
    sorted_df = sorted_df.sort_values(by=["date", "match_id"]).reset_index(drop=True)

    team_history = defaultdict(list)
    h2h_history = defaultdict(list)
    venue_history = defaultdict(list)

    records = []

    for _, row in sorted_df.iterrows():
        t1 = row["team1"]
        t2 = row["team2"]
        ven = row["venue"]
        winner = row["winner"]
        toss_win = row["toss_winner"]
        toss_dec = row["toss_decision"]

        # 1. Rolling Form (Last 5 & 10 matches)
        t1_hist = team_history[t1]
        t2_hist = team_history[t2]

        t1_form_5 = sum(t1_hist[-5:]) / len(t1_hist[-5:]) if len(t1_hist[-5:]) > 0 else 0.5
        t2_form_5 = sum(t2_hist[-5:]) / len(t2_hist[-5:]) if len(t2_hist[-5:]) > 0 else 0.5
        t1_form_10 = sum(t1_hist[-10:]) / len(t1_hist[-10:]) if len(t1_hist[-10:]) > 0 else 0.5
        t2_form_10 = sum(t2_hist[-10:]) / len(t2_hist[-10:]) if len(t2_hist[-10:]) > 0 else 0.5

        # Career prior win rate up to this match date
        t1_win_rate = sum(t1_hist) / len(t1_hist) if len(t1_hist) > 0 else 0.5
        t2_win_rate = sum(t2_hist) / len(t2_hist) if len(t2_hist) > 0 else 0.5

        # 2. Head-to-Head win rate
        pair_key = tuple(sorted([t1, t2]))
        pair_hist = h2h_history[pair_key]
        if len(pair_hist) > 0:
            t1_h2h_wins = sum(1 for w in pair_hist if w == t1)
            t1_h2h_win_rate = t1_h2h_wins / len(pair_hist)
        else:
            t1_h2h_win_rate = 0.5
        h2h_matches = len(pair_hist)

        # 3. Venue win rate
        t1_ven_hist = venue_history[(t1, ven)]
        t2_ven_hist = venue_history[(t2, ven)]
        t1_ven_rate = sum(t1_ven_hist) / len(t1_ven_hist) if len(t1_ven_hist) > 0 else t1_win_rate
        t2_ven_rate = sum(t2_ven_hist) / len(t2_ven_hist) if len(t2_ven_hist) > 0 else t2_win_rate

        # 4. Toss & Home/Away flags
        t1_won_toss = 1 if toss_win == t1 else 0
        t1_bat_first = 1 if ((t1_won_toss == 1 and toss_dec == "bat") or (t1_won_toss == 0 and toss_dec == "field")) else 0
        t1_is_home = 1 if ven in FRANCHISE_HOME_VENUES.get(t1, []) else 0
        t2_is_home = 1 if ven in FRANCHISE_HOME_VENUES.get(t2, []) else 0

        # 5. Target variable
        team1_won = 1 if winner == t1 else 0

        feat = {
            "match_id": row["match_id"],
            "season": row["season"],
            "date": row["date"],
            "venue": ven,
            "city": row.get("city", "Unknown"),
            "team1": t1,
            "team2": t2,
            "toss_winner": toss_win,
            "toss_decision": toss_dec,
            "team1_won_toss": t1_won_toss,
            "team1_batting_first": t1_bat_first,
            "team1_is_home": t1_is_home,
            "team2_is_home": t2_is_home,
            "team1_form_5": round(t1_form_5, 4),
            "team2_form_5": round(t2_form_5, 4),
            "team1_form_10": round(t1_form_10, 4),
            "team2_form_10": round(t2_form_10, 4),
            "form_diff_5": round(t1_form_5 - t2_form_5, 4),
            "team1_overall_win_rate": round(t1_win_rate, 4),
            "team2_overall_win_rate": round(t2_win_rate, 4),
            "overall_win_rate_diff": round(t1_win_rate - t2_win_rate, 4),
            "h2h_matches": h2h_matches,
            "team1_h2h_win_rate": round(t1_h2h_win_rate, 4),
            "team1_venue_win_rate": round(t1_ven_rate, 4),
            "team2_venue_win_rate": round(t2_ven_rate, 4),
            "venue_win_rate_diff": round(t1_ven_rate - t2_ven_rate, 4),
            "winner": winner,
            "team1_won": team1_won,
        }
        records.append(feat)

        # Update historical state AFTER calculating match features (strictly prevents data leakage)
        team_history[t1].append(1 if winner == t1 else 0)
        team_history[t2].append(1 if winner == t2 else 0)
        h2h_history[pair_key].append(winner)
        venue_history[(t1, ven)].append(1 if winner == t1 else 0)
        venue_history[(t2, ven)].append(1 if winner == t2 else 0)

    features_df = pd.DataFrame(records)
    return features_df


def run_feature_engineering():
    """Load cleaned matches, compute features, save features.csv, and report schema."""
    candidate_paths = [
        os.path.abspath("data/processed/matches_clean.csv"),
        os.path.abspath("cricket-match-prediction/data/processed/matches_clean.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "matches_clean.csv"),
    ]

    clean_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            clean_path = p
            break

    if not clean_path:
        raise FileNotFoundError(f"Cannot find matches_clean.csv in: {candidate_paths}")

    print(f"Loading cleaned dataset from: {clean_path}")
    clean_df = pd.read_csv(clean_path)

    print("Engineering predictive features...")
    features_df = engineer_features(clean_df)

    output_locations = [
        os.path.abspath("data/processed/features.csv"),
        os.path.abspath("cricket-match-prediction/data/processed/features.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv"),
    ]

    saved_paths = set()
    for out_path in output_locations:
        norm_path = os.path.normpath(out_path)
        if norm_path not in saved_paths:
            os.makedirs(os.path.dirname(norm_path), exist_ok=True)
            features_df.to_csv(norm_path, index=False)
            saved_paths.add(norm_path)
            print(f"Saved feature dataset to: {norm_path}")

    # Step 5 summary report
    print("\n" + "=" * 60)
    print("STEP 5 FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Final Features Shape: {features_df.shape} ({features_df.shape[0]} rows, {features_df.shape[1]} columns)")
    print("\nFinal Feature List:")
    for idx, col in enumerate(features_df.columns, 1):
        dtype = str(features_df[col].dtype)
        sample = features_df[col].iloc[-1]
        print(f"  {idx:2d}. {col:<26} (dtype: {dtype:<10}, sample: {sample})")

    print("\nTarget Class Distribution ('team1_won'):")
    print(features_df["team1_won"].value_counts().to_string())
    print("=" * 60)

    return features_df


if __name__ == "__main__":
    run_feature_engineering()
