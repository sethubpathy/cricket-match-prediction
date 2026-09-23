"""Streamlit Web Application for Cricket Match Outcome Prediction."""

import os
import sys
import pandas as pd
import streamlit as st

# Ensure project directories are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
subfolder_src = os.path.join(current_dir, "cricket-match-prediction")
if os.path.exists(subfolder_src) and subfolder_src not in sys.path:
    sys.path.insert(0, subfolder_src)

from src.predict import predict_match, load_clean_data

# Page Configuration
st.set_page_config(
    page_title="Cricket Match Outcome Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (CSS)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #4b5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    .winner-card {
        background: linear-gradient(135deg, #1e1e38 0%, #2a2b58 100%);
        color: #ffffff;
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .winner-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .winner-name {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0.4rem 0;
        letter-spacing: -0.02em;
    }
    
    .confidence-pill {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Reference Data
@st.cache_data
def get_dropdown_options():
    df = load_clean_data()
    all_teams = sorted(list(set(df["team1"]).union(set(df["team2"]))))
    
    # Core active franchises
    active_teams = [
        "Chennai Super Kings",
        "Mumbai Indians",
        "Royal Challengers Bengaluru",
        "Kolkata Knight Riders",
        "Delhi Capitals",
        "Gujarat Titans",
        "Lucknow Super Giants",
        "Punjab Kings",
        "Rajasthan Royals",
        "Sunrisers Hyderabad",
    ]
    historical_teams = [t for t in all_teams if t not in active_teams]
    ordered_teams = active_teams + historical_teams

    all_venues = sorted(df["venue"].unique().tolist())
    top_venues = [
        "Wankhede Stadium",
        "MA Chidambaram Stadium",
        "M Chinnaswamy Stadium",
        "Eden Gardens",
        "Narendra Modi Stadium, Ahmedabad",
        "Arun Jaitley Stadium",
        "Rajiv Gandhi International Stadium",
        "Punjab Cricket Association IS Bindra Stadium",
        "Sawai Mansingh Stadium",
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    ]
    other_venues = [v for v in all_venues if v not in top_venues]
    ordered_venues = top_venues + other_venues

    return ordered_teams, ordered_venues

teams_list, venues_list = get_dropdown_options()

# Header
st.markdown('<div class="main-title">🏏 Cricket Match Outcome Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict the match winner and calibrated win probability using historical match data and a tuned RandomForestClassifier.</div>',
    unsafe_allow_html=True,
)

# Initialize session state defaults
DEFAULT_T1 = "Chennai Super Kings"
DEFAULT_T2 = "Mumbai Indians"
DEFAULT_VEN = "Wankhede Stadium"
DEFAULT_TW = "Chennai Super Kings"
DEFAULT_TD = "field"

if "sb_t1" not in st.session_state:
    st.session_state["sb_t1"] = DEFAULT_T1
if "sb_t2" not in st.session_state:
    st.session_state["sb_t2"] = DEFAULT_T2
if "sb_ven" not in st.session_state:
    st.session_state["sb_ven"] = DEFAULT_VEN
if "sb_tw" not in st.session_state:
    st.session_state["sb_tw"] = DEFAULT_TW
if "rb_td" not in st.session_state:
    st.session_state["rb_td"] = DEFAULT_TD
if "preset_status_msg" not in st.session_state:
    st.session_state["preset_status_msg"] = ""

# Callback functions for robust state mutation
def apply_preset(team1, team2, venue, toss_winner, toss_decision, matchup_name):
    st.session_state["sb_t1"] = team1
    st.session_state["sb_t2"] = team2
    st.session_state["sb_ven"] = venue
    st.session_state["sb_tw"] = toss_winner
    st.session_state["rb_td"] = toss_decision
    st.session_state["preset_status_msg"] = f"⚡ Loaded preset **{matchup_name}**! Match parameters updated below."

def on_team1_change():
    t1 = st.session_state.get("sb_t1")
    t2 = st.session_state.get("sb_t2")
    if t1 == t2:
        for t in teams_list:
            if t != t1:
                st.session_state["sb_t2"] = t
                break
    t2_cur = st.session_state.get("sb_t2")
    tw = st.session_state.get("sb_tw")
    if tw not in [t1, t2_cur]:
        st.session_state["sb_tw"] = t1
    st.session_state["preset_status_msg"] = ""

def on_team2_change():
    t1 = st.session_state.get("sb_t1")
    t2 = st.session_state.get("sb_t2")
    tw = st.session_state.get("sb_tw")
    if tw not in [t1, t2]:
        st.session_state["sb_tw"] = t1
    st.session_state["preset_status_msg"] = ""

# Quick Preset Buttons
st.markdown("##### ⚡ Quick Match Presets")
preset_col1, preset_col2, preset_col3 = st.columns(3)

with preset_col1:
    st.button(
        "🔥 MI vs CSK (El Clásico)",
        use_container_width=True,
        key="btn_preset_1",
        on_click=apply_preset,
        args=("Chennai Super Kings", "Mumbai Indians", "Wankhede Stadium", "Chennai Super Kings", "field", "MI vs CSK"),
    )

with preset_col2:
    st.button(
        "⚔️ KKR vs RCB (Royal Derby)",
        use_container_width=True,
        key="btn_preset_2",
        on_click=apply_preset,
        args=("Kolkata Knight Riders", "Royal Challengers Bengaluru", "Eden Gardens", "Kolkata Knight Riders", "field", "KKR vs RCB"),
    )

with preset_col3:
    st.button(
        "🛡️ GT vs RR (Finalist Rematch)",
        use_container_width=True,
        key="btn_preset_3",
        on_click=apply_preset,
        args=("Gujarat Titans", "Rajasthan Royals", "Narendra Modi Stadium, Ahmedabad", "Gujarat Titans", "field", "GT vs RR"),
    )

if st.session_state.get("preset_status_msg"):
    st.info(st.session_state["preset_status_msg"])

st.markdown("---")

# Main Input Layout
left_col, right_col = st.columns([1.1, 1.4], gap="large")

with left_col:
    st.markdown("### ⚙️ Match Parameters")
    
    # Team 1 selection
    selected_t1 = st.selectbox(
        "Select Team 1",
        options=teams_list,
        key="sb_t1",
        on_change=on_team1_change,
    )

    # Team 2 selection (cannot be same as Team 1)
    available_t2 = [t for t in teams_list if t != selected_t1]
    if st.session_state.get("sb_t2") not in available_t2:
        st.session_state["sb_t2"] = available_t2[0]

    selected_t2 = st.selectbox(
        "Select Team 2",
        options=available_t2,
        key="sb_t2",
        on_change=on_team2_change,
    )

    # Venue selection
    selected_venue = st.selectbox(
        "Select Match Venue",
        options=venues_list,
        key="sb_ven",
    )

    # Toss Winner
    toss_candidates = [selected_t1, selected_t2]
    if st.session_state.get("sb_tw") not in toss_candidates:
        st.session_state["sb_tw"] = toss_candidates[0]

    selected_toss_winner = st.selectbox(
        "Toss Winner",
        options=toss_candidates,
        key="sb_tw",
    )

    # Toss Decision
    selected_toss_decision = st.radio(
        "Toss Decision",
        options=["field", "bat"],
        key="rb_td",
        format_func=lambda x: "Field (Bowling First / Chasing)" if x == "field" else "Bat (Batting First / Defending)",
        horizontal=True,
    )

    predict_btn = st.button("🔮 Predict Match Winner", type="primary", use_container_width=True, key="btn_predict")

with right_col:
    st.markdown("### 📊 Prediction & Matchup Analytics")
    
    current_params = (selected_t1, selected_t2, selected_venue, selected_toss_winner, selected_toss_decision)
    
    # Calculate or update prediction on predict button click or initial mount
    if "prediction_result" not in st.session_state or predict_btn:
        with st.spinner("Analyzing matchup telemetry and running prediction..."):
            st.session_state["prediction_result"] = predict_match(
                team1=selected_t1,
                team2=selected_t2,
                venue=selected_venue,
                toss_winner=selected_toss_winner,
                toss_decision=selected_toss_decision,
            )
            st.session_state["prediction_params"] = current_params
            st.session_state["preset_status_msg"] = ""
            
        if predict_btn:
            st.toast(
                f"Prediction updated: {st.session_state['prediction_result']['predicted_winner']} ({st.session_state['prediction_result']['win_probability']:.1f}%)",
                icon="🏏",
            )
            st.success(f"✅ Prediction updated for **{selected_t1}** vs. **{selected_t2}** at **{selected_venue.split(',')[0]}**!")

    res = st.session_state["prediction_result"]
    saved_params = st.session_state.get("prediction_params")
    
    # Alert if parameters changed since last prediction calculation
    if saved_params and saved_params != current_params:
        st.warning("⚠️ Match parameters have changed. Click **'🔮 Predict Match Winner'** to update this forecast.")
    
    winner = res["predicted_winner"]
    win_prob = res["win_probability"]
    t1_prob = res["team1_win_probability"]
    t2_prob = res["team2_win_probability"]
    feat = res["features"]

    # Winner Card
    st.markdown(
        f"""
        <div class="winner-card">
            <div class="winner-badge">Model Predicted Outcome</div>
            <div class="winner-name">🏆 {winner}</div>
            <div class="confidence-pill">Win Probability: {win_prob:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Win Probability Breakdown Bar
    st.markdown("##### 📈 Head-to-Head Win Probability")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.write(f"**{selected_t1}**")
        st.write(f"**{t1_prob:.1f}%**")
    with p_col2:
        st.write(f"<div style='text-align: right;'><b>{selected_t2}</b><br><b>{t2_prob:.1f}%</b></div>", unsafe_allow_html=True)
    
    st.progress(float(t1_prob / 100.0))

    st.markdown("---")
    st.markdown("##### 🔍 Contextual Match Intelligence")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Head-to-Head</div>
                <div class="metric-val">{feat['h2h_matches']} Games</div>
                <div style="font-size: 0.8rem; color: #6b7280; margin-top: 4px;">{selected_t1} Win Rate: {feat['team1_h2h_win_rate']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{selected_t1} Form</div>
                <div class="metric-val">{feat['team1_form_5']*100:.0f}%</div>
                <div style="font-size: 0.8rem; color: #6b7280; margin-top: 4px;">Last 5 Games (10G: {feat['team1_form_10']*100:.0f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{selected_t2} Form</div>
                <div class="metric-val">{feat['team2_form_5']*100:.0f}%</div>
                <div style="font-size: 0.8rem; color: #6b7280; margin-top: 4px;">Last 5 Games (10G: {feat['team2_form_10']*100:.0f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{selected_t1} @ {selected_venue.split(',')[0]}</div>
                <div class="metric-val">{feat['team1_venue_win_rate']*100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #6b7280; margin-top: 4px;">{'Home Ground Advantage' if feat['team1_is_home'] else 'Away Ground'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{selected_t2} @ {selected_venue.split(',')[0]}</div>
                <div class="metric-val">{feat['team2_venue_win_rate']*100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #6b7280; margin-top: 4px;">{'Home Ground Advantage' if feat['team2_is_home'] else 'Away Ground'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption("Cricket Match Outcome Prediction System • Built with Python, Pandas, Scikit-Learn & Streamlit.")
