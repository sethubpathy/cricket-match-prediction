"""Match prediction module and CLI interface for Cricket Match Outcome Prediction."""

import os
import argparse
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np

# Team and Venue mappings for robust input handling
TEAM_MAPPINGS = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "RCB": "Royal Challengers Bengaluru",
    "CSK": "Chennai Super Kings",
    "MI": "Mumbai Indians",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "PBKS": "Punjab Kings",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
}

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

CATEGORICAL_FEATURES = ["team1", "team2", "venue", "toss_decision"]
NUMERIC_FEATURES = [
    "team1_won_toss",
    "team1_batting_first",
    "team1_is_home",
    "team2_is_home",
    "team1_form_5",
    "team2_form_5",
    "team1_form_10",
    "team2_form_10",
    "form_diff_5",
    "team1_overall_win_rate",
    "team2_overall_win_rate",
    "overall_win_rate_diff",
    "h2h_matches",
    "team1_h2h_win_rate",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
    "venue_win_rate_diff",
]


def load_model(model_path: Optional[str] = None):
    """Load serialized pipeline with fallback path resolution."""
    if model_path and os.path.exists(model_path):
        return joblib.load(model_path)

    candidates = [
        os.path.abspath("models/cricket_model.pkl"),
        os.path.abspath("cricket-match-prediction/models/cricket_model.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "models", "cricket_model.pkl"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p)

    raise FileNotFoundError(f"Model file not found in candidate paths: {candidates}")


def load_clean_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load cleaned matches CSV for on-the-fly historical state lookups."""
    if data_path and os.path.exists(data_path):
        return pd.read_csv(data_path)

    candidates = [
        os.path.abspath("data/processed/matches_clean.csv"),
        os.path.abspath("cricket-match-prediction/data/processed/matches_clean.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "matches_clean.csv"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)

    raise FileNotFoundError(f"Matches dataset not found in candidate paths: {candidates}")


def build_feature_row(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    matches_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the exact 21-feature row on the fly from current inputs and historical records."""
    if matches_df is None:
        matches_df = load_clean_data()

    # Standardize names
    t1 = TEAM_MAPPINGS.get(team1.strip(), team1.strip())
    t2 = TEAM_MAPPINGS.get(team2.strip(), team2.strip())
    tw = TEAM_MAPPINGS.get(toss_winner.strip(), toss_winner.strip())
    ven = VENUE_MAPPINGS.get(venue.strip(), venue.strip())
    tdec = toss_decision.strip().lower()

    if tdec not in ["bat", "field"]:
        tdec = "field"

    # 1. Historical Matches for Team 1 & Team 2
    t1_matches = matches_df[(matches_df["team1"] == t1) | (matches_df["team2"] == t1)]
    t2_matches = matches_df[(matches_df["team1"] == t2) | (matches_df["team2"] == t2)]

    t1_wins = (t1_matches["winner"] == t1).tolist()
    t2_wins = (t2_matches["winner"] == t2).tolist()

    # Rolling form (last 5 & 10)
    t1_form_5 = sum(t1_wins[-5:]) / len(t1_wins[-5:]) if len(t1_wins[-5:]) > 0 else 0.5
    t2_form_5 = sum(t2_wins[-5:]) / len(t2_wins[-5:]) if len(t2_wins[-5:]) > 0 else 0.5
    t1_form_10 = sum(t1_wins[-10:]) / len(t1_wins[-10:]) if len(t1_wins[-10:]) > 0 else 0.5
    t2_form_10 = sum(t2_wins[-10:]) / len(t2_wins[-10:]) if len(t2_wins[-10:]) > 0 else 0.5

    # Overall career win rate
    t1_overall = sum(t1_wins) / len(t1_wins) if len(t1_wins) > 0 else 0.5
    t2_overall = sum(t2_wins) / len(t2_wins) if len(t2_wins) > 0 else 0.5

    # 2. Head-to-Head (H2H)
    h2h_df = matches_df[
        ((matches_df["team1"] == t1) & (matches_df["team2"] == t2))
        | ((matches_df["team1"] == t2) & (matches_df["team2"] == t1))
    ]
    h2h_matches = len(h2h_df)
    if h2h_matches > 0:
        t1_h2h_wins = int((h2h_df["winner"] == t1).sum())
        t1_h2h_rate = t1_h2h_wins / h2h_matches
    else:
        t1_h2h_rate = 0.5

    # 3. Venue Win Rate
    t1_venue_df = matches_df[
        ((matches_df["team1"] == t1) | (matches_df["team2"] == t1))
        & (matches_df["venue"] == ven)
    ]
    t2_venue_df = matches_df[
        ((matches_df["team1"] == t2) | (matches_df["team2"] == t2))
        & (matches_df["venue"] == ven)
    ]

    t1_venue_rate = (
        (t1_venue_df["winner"] == t1).mean() if len(t1_venue_df) > 0 else t1_overall
    )
    t2_venue_rate = (
        (t2_venue_df["winner"] == t2).mean() if len(t2_venue_df) > 0 else t2_overall
    )

    # 4. Toss & Home/Away Flags
    t1_won_toss = 1 if tw == t1 else 0
    t1_bat_first = 1 if ((t1_won_toss == 1 and tdec == "bat") or (t1_won_toss == 0 and tdec == "field")) else 0
    t1_is_home = 1 if ven in FRANCHISE_HOME_VENUES.get(t1, []) else 0
    t2_is_home = 1 if ven in FRANCHISE_HOME_VENUES.get(t2, []) else 0

    feature_dict = {
        "team1": t1,
        "team2": t2,
        "venue": ven,
        "toss_decision": tdec,
        "team1_won_toss": t1_won_toss,
        "team1_batting_first": t1_bat_first,
        "team1_is_home": t1_is_home,
        "team2_is_home": t2_is_home,
        "team1_form_5": round(t1_form_5, 4),
        "team2_form_5": round(t2_form_5, 4),
        "team1_form_10": round(t1_form_10, 4),
        "team2_form_10": round(t2_form_10, 4),
        "form_diff_5": round(t1_form_5 - t2_form_5, 4),
        "team1_overall_win_rate": round(t1_overall, 4),
        "team2_overall_win_rate": round(t2_overall, 4),
        "overall_win_rate_diff": round(t1_overall - t2_overall, 4),
        "h2h_matches": h2h_matches,
        "team1_h2h_win_rate": round(t1_h2h_rate, 4),
        "team1_venue_win_rate": round(t1_venue_rate, 4),
        "team2_venue_win_rate": round(t2_venue_rate, 4),
        "venue_win_rate_diff": round(t1_venue_rate - t2_venue_rate, 4),
    }

    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    row_df = pd.DataFrame([feature_dict])[cols]
    return row_df


def predict_match(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    model=None,
    matches_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Exposed prediction function returning predicted winner and win probability."""
    if model is None:
        model = load_model()
    if matches_df is None:
        matches_df = load_clean_data()

    # Build feature row on the fly
    feature_row = build_feature_row(
        team1, team2, venue, toss_winner, toss_decision, matches_df=matches_df
    )

    # Class 1: team1 won, Class 0: team2 won
    prob_distribution = model.predict_proba(feature_row)[0]
    class_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1

    team1_prob = float(prob_distribution[class_idx])
    team2_prob = float(1.0 - team1_prob)

    norm_team1 = feature_row["team1"].iloc[0]
    norm_team2 = feature_row["team2"].iloc[0]

    if team1_prob >= 0.50:
        predicted_winner = norm_team1
        win_probability = team1_prob
    else:
        predicted_winner = norm_team2
        win_probability = team2_prob

    result = {
        "team1": norm_team1,
        "team2": norm_team2,
        "venue": feature_row["venue"].iloc[0],
        "toss_winner": feature_row["team1"].iloc[0] if feature_row["team1_won_toss"].iloc[0] == 1 else norm_team2,
        "toss_decision": feature_row["toss_decision"].iloc[0],
        "predicted_winner": predicted_winner,
        "win_probability": round(win_probability * 100, 2),
        "team1_win_probability": round(team1_prob * 100, 2),
        "team2_win_probability": round(team2_prob * 100, 2),
        "features": feature_row.iloc[0].to_dict(),
    }

    return result


def print_prediction_card(res: Dict[str, Any]):
    """Print an attractive, formatted CLI prediction summary card using console-safe ASCII."""
    t1 = res["team1"]
    t2 = res["team2"]
    t1_pct = res["team1_win_probability"]
    t2_pct = res["team2_win_probability"]
    feat = res["features"]

    # Safe ASCII probability bar (25 characters total width)
    t1_bar_len = int(round(t1_pct / 4))
    t2_bar_len = 25 - t1_bar_len
    bar = "#" * t1_bar_len + "-" * t2_bar_len

    border = "=" * 68
    sub_border = "-" * 68

    print("\n" + border)
    print(f"{'CRICKET MATCH OUTCOME PREDICTION':^68}")
    print(border)
    print(f" Matchup:       {t1} vs. {t2}")
    print(f" Venue:         {res['venue']}")
    print(f" Toss:          {res['toss_winner']} (Elected to {res['toss_decision'].upper()})")
    print(sub_border)
    print(f" PREDICTED WINNER:  {res['predicted_winner'].upper()}")
    print(f" WIN CONFIDENCE:    {res['win_probability']:.1f}%")
    print(sub_border)
    print(" Win Probability Breakdown:")
    print(f"   * {t1:<28}: {t1_pct:>5.1f}%")
    print(f"   * {t2:<28}: {t2_pct:>5.1f}%")
    print(f"   [{bar}] {t1_pct:.1f}% vs {t2_pct:.1f}%")
    print(sub_border)
    print(" Matchup Analytics Context:")
    print(f"   * Head-to-Head History:  {feat['h2h_matches']} prior games ({t1} win rate: {feat['team1_h2h_win_rate']*100:.1f}%)")
    print(f"   * Recent Form (5G):      {t1}: {feat['team1_form_5']*100:.0f}%  |  {t2}: {feat['team2_form_5']*100:.0f}%")
    print(f"   * Venue Track Record:    {t1}: {feat['team1_venue_win_rate']*100:.1f}%  |  {t2}: {feat['team2_venue_win_rate']*100:.1f}%")
    print(border + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict cricket match winner and win probability using tuned RandomForest pipeline."
    )
    parser.add_argument(
        "--team1",
        type=str,
        default="Chennai Super Kings",
        help="Name of Team 1 (default: 'Chennai Super Kings')",
    )
    parser.add_argument(
        "--team2",
        type=str,
        default="Mumbai Indians",
        help="Name of Team 2 (default: 'Mumbai Indians')",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default="Wankhede Stadium",
        help="Match Venue / Stadium (default: 'Wankhede Stadium')",
    )
    parser.add_argument(
        "--toss_winner",
        type=str,
        default="Chennai Super Kings",
        help="Team that won the coin toss (default: 'Chennai Super Kings')",
    )
    parser.add_argument(
        "--toss_decision",
        type=str,
        choices=["bat", "field"],
        default="field",
        help="Decision upon winning toss: 'bat' or 'field' (default: 'field')",
    )

    args = parser.parse_args()

    res = predict_match(
        team1=args.team1,
        team2=args.team2,
        venue=args.venue,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
    )

    print_prediction_card(res)


if __name__ == "__main__":
    main()
