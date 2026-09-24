"""Streamlit Web Application for Cricket Match Outcome Prediction.

A modern, responsive sports-analytics dashboard predicting IPL match winners
and calibrated win probabilities using historical data and a tuned RandomForestClassifier.
"""

import os
import sys
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

# Ensure project directories are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
subfolder_src = os.path.join(current_dir, "cricket-match-prediction")
if os.path.exists(subfolder_src) and subfolder_src not in sys.path:
    sys.path.insert(0, subfolder_src)

from src.predict import predict_match, load_clean_data, load_model
from src.ui_components import (
    TEAM_METADATA,
    get_team_info,
    get_team_recent_results,
    create_win_gauge,
    create_h2h_chart,
    create_form_comparison_chart,
    create_venue_comparison_chart,
    generate_ai_summary,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cricket Match Outcome Predictor | IPL AI Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# 2. Modern Sports-Analytics CSS Design System
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F8FAFC;
}

/* App Background & Canvas */
.stApp {
    background: radial-gradient(circle at 50% 0%, #0F172A 0%, #070D19 100%);
    background-attachment: fixed;
}

/* Hero Header */
.hero-container {
    text-align: center;
    padding: 1.8rem 1rem 1.2rem 1rem;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #FFFFFF 0%, #93C5FD 50%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 720px;
    margin: 0 auto 1.2rem auto;
    line-height: 1.5;
}

.badge-row {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}

.model-badge {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #E2E8F0;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.3rem 0.8rem;
    border-radius: 9999px;
    letter-spacing: 0.02em;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

/* Surface Cards */
.glass-card {
    background: #0F172A;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.4rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    margin-bottom: 1.2rem;
}

.card-header-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Winner Card */
.winner-hero-card {
    border-radius: 20px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.6);
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.winner-sub-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    margin-bottom: 0.75rem;
}

.winner-franchise-name {
    font-size: 2.3rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin: 0.2rem 0 0.6rem 0;
    line-height: 1.15;
}

.confidence-chip {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.35rem 0.9rem;
    border-radius: 9999px;
    letter-spacing: 0.03em;
}

/* Custom H2H Win Probability Bar */
.h2h-prob-wrapper {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
}

.h2h-team-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
    font-weight: 700;
}

.h2h-team-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.95rem;
}

.h2h-bar-container {
    height: 16px;
    width: 100%;
    background: #1E293B;
    border-radius: 9999px;
    overflow: hidden;
    display: flex;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
}

.h2h-bar-left {
    height: 100%;
    transition: width 0.6s ease-in-out;
}

.h2h-bar-right {
    height: 100%;
    transition: width 0.6s ease-in-out;
}

/* Form W/L Badges */
.wl-badge-row {
    display: flex;
    gap: 0.35rem;
    align-items: center;
    margin-top: 0.4rem;
}

.wl-dot {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 800;
    color: #FFFFFF;
}

.wl-dot.win {
    background: #10B981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}

.wl-dot.loss {
    background: #EF4444;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
}

/* Context Metrics Grid */
.metric-grid-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 0.95rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.metric-grid-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

.metric-grid-val {
    font-size: 1.35rem;
    font-weight: 800;
    color: #F8FAFC;
}

.metric-grid-sub {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 0.2rem;
}

/* AI Summary Box */
.ai-summary-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-left: 4px solid #38BDF8;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #CBD5E1;
}

/* Footer */
.footer-container {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    color: #64748B;
    font-size: 0.82rem;
    margin-top: 2.5rem;
}

.footer-container a {
    color: #38BDF8;
    text-decoration: none;
    font-weight: 600;
}

.footer-container a:hover {
    text-decoration: underline;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .hero-title {
        font-size: 1.9rem;
    }
    .winner-franchise-name {
        font-size: 1.7rem;
    }
    .badge-row {
        gap: 0.35rem;
    }
    .model-badge {
        font-size: 0.68rem;
        padding: 0.25rem 0.6rem;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Data & Resource Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_matches_data() -> pd.DataFrame:
    """Load and cache the cleaned matches dataset."""
    return load_clean_data()


@st.cache_resource
def get_cached_model():
    """Load and cache the serialized scikit-learn pipeline."""
    return load_model()


@st.cache_data
def get_dropdown_options():
    """Build standardized, franchise-curated team and venue lists."""
    df = get_cached_matches_data()
    all_teams = sorted(list(set(df["team1"]).union(set(df["team2"]))))

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


matches_df = get_cached_matches_data()
model_pipeline = get_cached_model()
teams_list, venues_list = get_dropdown_options()


# -----------------------------------------------------------------------------
# 4. Session State & Preset Management
# -----------------------------------------------------------------------------
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


def apply_preset(team1: str, team2: str, venue: str, toss_winner: str, toss_decision: str, name: str):
    """Callback to cleanly apply quick preset values into widget state."""
    st.session_state["sb_t1"] = team1
    st.session_state["sb_t2"] = team2
    st.session_state["sb_ven"] = venue
    st.session_state["sb_tw"] = toss_winner
    st.session_state["rb_td"] = toss_decision
    st.session_state["preset_status_msg"] = f"⚡ Loaded Preset: **{name}**. Click '🔮 Predict Match Outcome' to compute!"


def on_team1_change():
    """Ensure Team 2 does not duplicate Team 1 and Toss Winner remains valid."""
    t1 = st.session_state.get("sb_t1")
    t2 = st.session_state.get("sb_t2")
    if t1 == t2:
        for team in teams_list:
            if team != t1:
                st.session_state["sb_t2"] = team
                break
    t2_current = st.session_state.get("sb_t2")
    tw = st.session_state.get("sb_tw")
    if tw not in [t1, t2_current]:
        st.session_state["sb_tw"] = t1
    st.session_state["preset_status_msg"] = ""


def on_team2_change():
    """Ensure Toss Winner candidates align with Team 1 & Team 2."""
    t1 = st.session_state.get("sb_t1")
    t2 = st.session_state.get("sb_t2")
    tw = st.session_state.get("sb_tw")
    if tw not in [t1, t2]:
        st.session_state["sb_tw"] = t1
    st.session_state["preset_status_msg"] = ""


# -----------------------------------------------------------------------------
# 5. Header Component
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🏏 Cricket Match Outcome Predictor</div>
        <div class="hero-subtitle">
            AI-powered Indian Premier League outcome forecasting & win probability engine built with historical match telemetry and a regularized machine learning ensemble.
        </div>
        <div class="badge-row">
            <span class="model-badge">⚡ RandomForestClassifier (Tuned)</span>
            <span class="model-badge">📊 17 IPL Seasons (2008–2024)</span>
            <span class="model-badge">🎯 28 Predictive Telemetry Features</span>
            <span class="model-badge">⏱️ Time-Aware Split Validation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 6. Quick Match Presets
# -----------------------------------------------------------------------------
st.markdown("##### ⚡ Quick Match Presets")
preset_c1, preset_c2, preset_c3 = st.columns(3)

with preset_c1:
    st.button(
        "🔥 MI vs CSK (El Clásico)",
        use_container_width=True,
        key="btn_preset_1",
        on_click=apply_preset,
        args=("Chennai Super Kings", "Mumbai Indians", "Wankhede Stadium", "Chennai Super Kings", "field", "MI vs CSK"),
    )

with preset_c2:
    st.button(
        "⚔️ KKR vs RCB (Royal Derby)",
        use_container_width=True,
        key="btn_preset_2",
        on_click=apply_preset,
        args=("Kolkata Knight Riders", "Royal Challengers Bengaluru", "Eden Gardens", "Kolkata Knight Riders", "field", "KKR vs RCB"),
    )

with preset_c3:
    st.button(
        "🛡️ GT vs RR (Finalist Rematch)",
        use_container_width=True,
        key="btn_preset_3",
        on_click=apply_preset,
        args=("Gujarat Titans", "Rajasthan Royals", "Narendra Modi Stadium, Ahmedabad", "Gujarat Titans", "field", "GT vs RR"),
    )

if st.session_state.get("preset_status_msg"):
    st.info(st.session_state["preset_status_msg"])

st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Navigation Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_analytics, tab_model, tab_about = st.tabs([
    "🔮 Match Predictor",
    "📊 Matchup Deep-Dive",
    "🧠 Model Intelligence",
    "ℹ️ Data & Methodology",
])


# =============================================================================
# TAB 1: Core Predictor & Analytics
# =============================================================================
with tab_predict:
    col_input, col_results = st.columns([1.05, 1.4], gap="large")

    # ------------------ LEFT COLUMN: PARAMETERS ------------------
    with col_input:
        st.markdown(
            """
            <div class="card-header-title">
                <span>⚙️ Match Parameters</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_t1 = st.selectbox(
            "Select Team 1",
            options=teams_list,
            key="sb_t1",
            on_change=on_team1_change,
        )

        available_t2 = [t for t in teams_list if t != selected_t1]
        if st.session_state.get("sb_t2") not in available_t2:
            st.session_state["sb_t2"] = available_t2[0]

        selected_t2 = st.selectbox(
            "Select Team 2",
            options=available_t2,
            key="sb_t2",
            on_change=on_team2_change,
        )

        selected_venue = st.selectbox(
            "Match Venue",
            options=venues_list,
            key="sb_ven",
        )

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

        st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True, key="btn_predict")

    # ------------------ RIGHT COLUMN: RESULTS ------------------
    with col_results:
        current_params = (selected_t1, selected_t2, selected_venue, selected_toss_winner, selected_toss_decision)

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
                    f"Forecast updated: {st.session_state['prediction_result']['predicted_winner']} ({st.session_state['prediction_result']['win_probability']:.1f}%)",
                    icon="🏏",
                )

        res = st.session_state["prediction_result"]
        saved_params = st.session_state.get("prediction_params")

        # Alert if inputs changed since last prediction calculation
        if saved_params and saved_params != current_params:
            st.warning("⚠️ Match parameters have changed. Click **'🔮 Predict Match Outcome'** to calculate the new forecast.")

        winner = res["predicted_winner"]
        win_prob = res["win_probability"]
        t1_prob = res["team1_win_probability"]
        t2_prob = res["team2_win_probability"]
        feat = res["features"]

        # Team metadata
        t1_info = get_team_info(selected_t1)
        t2_info = get_team_info(selected_t2)
        winner_info = get_team_info(winner)

        # AI summary & confidence label
        conf_label, badge_style, ai_summary_text = generate_ai_summary(res)

        # Winner Hero Card with franchise gradient
        winner_bg_gradient = f"linear-gradient(135deg, {winner_info['primary']}22 0%, #0F172A 75%, {winner_info['secondary']}15 100%)"
        st.markdown(
            f"""
            <div class="winner-hero-card" style="background: {winner_bg_gradient}; border: 1px solid {winner_info['primary']}55;">
                <div class="winner-sub-badge" style="background: {winner_info['primary']}25; color: {winner_info['primary']}; border: 1px solid {winner_info['primary']}55;">
                    Model Projected Winner
                </div>
                <div class="winner-franchise-name" style="color: #FFFFFF;">
                    {winner_info['emoji']} {winner}
                </div>
                <div style="margin-top: 0.5rem;">
                    <span class="confidence-chip" style="{badge_style}">
                        {conf_label} • {win_prob:.1f}% Probability
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Head-to-Head Custom Dual-Color Win Probability Bar
        st.markdown(
            f"""
            <div class="h2h-prob-wrapper">
                <div class="h2h-team-row">
                    <span class="h2h-team-label" style="color: {t1_info['primary']};">
                        <span>{t1_info['emoji']}</span>
                        <span>{selected_t1}</span>
                    </span>
                    <span style="font-size: 1.15rem; color: #F8FAFC;">{t1_prob:.1f}% vs. {t2_prob:.1f}%</span>
                    <span class="h2h-team-label" style="color: {t2_info['primary']};">
                        <span>{selected_t2}</span>
                        <span>{t2_info['emoji']}</span>
                    </span>
                </div>
                <div class="h2h-bar-container">
                    <div class="h2h-bar-left" style="width: {t1_prob}%; background: {t1_info['primary']};"></div>
                    <div class="h2h-bar-right" style="width: {t2_prob}%; background: {t2_info['primary']};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Circular Gauge Chart & AI Narrative Row
        gauge_col, summary_col = st.columns([1, 1.25])
        with gauge_col:
            gauge_fig = create_win_gauge(winner, win_prob, selected_t1, selected_t2, t1_prob, t2_prob)
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

        with summary_col:
            st.markdown(
                f"""
                <div class="ai-summary-card">
                    <div style="font-weight: 700; color: #38BDF8; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.35rem;">
                        <span>🧠 Tactical Model Analysis</span>
                    </div>
                    <div>{ai_summary_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ------------------ CONTEXTUAL MATCH INTELLIGENCE ------------------
        st.markdown("##### 🔍 Matchup Context & Historical Records")

        # Extract actual last 5 match results for W/L badges
        t1_results_5 = get_team_recent_results(matches_df, selected_t1, 5)
        t2_results_5 = get_team_recent_results(matches_df, selected_t2, 5)

        def render_wl_dots(results: List[str]) -> str:
            dots_html = "".join([f'<span class="wl-dot {"win" if r == "W" else "loss"}">{r}</span>' for r in results])
            return f'<div class="wl-badge-row">{dots_html}</div>'

        c_m1, c_m2, c_m3 = st.columns(3)

        with c_m1:
            st.markdown(
                f"""
                <div class="metric-grid-card">
                    <div>
                        <div class="metric-grid-label">Head-to-Head</div>
                        <div class="metric-grid-val">{feat['h2h_matches']} Games</div>
                    </div>
                    <div class="metric-grid-sub">
                        <b>{t1_info['short']}</b>: {feat['team1_h2h_win_rate']*100:.1f}% Win Rate
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_m2:
            st.markdown(
                f"""
                <div class="metric-grid-card">
                    <div>
                        <div class="metric-grid-label">{t1_info['short']} Recent Form</div>
                        <div class="metric-grid-val">{feat['team1_form_5']*100:.0f}%</div>
                    </div>
                    <div>
                        {render_wl_dots(t1_results_5)}
                        <div class="metric-grid-sub" style="margin-top: 4px;">Last 10 Games: {feat['team1_form_10']*100:.0f}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_m3:
            st.markdown(
                f"""
                <div class="metric-grid-card">
                    <div>
                        <div class="metric-grid-label">{t2_info['short']} Recent Form</div>
                        <div class="metric-grid-val">{feat['team2_form_5']*100:.0f}%</div>
                    </div>
                    <div>
                        {render_wl_dots(t2_results_5)}
                        <div class="metric-grid-sub" style="margin-top: 4px;">Last 10 Games: {feat['team2_form_10']*100:.0f}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        v_col1, v_col2 = st.columns(2)
        venue_short = selected_venue.split(",")[0]

        with v_col1:
            st.markdown(
                f"""
                <div class="metric-grid-card" style="margin-top: 0.6rem;">
                    <div>
                        <div class="metric-grid-label">{t1_info['short']} @ {venue_short}</div>
                        <div class="metric-grid-val">{feat['team1_venue_win_rate']*100:.1f}%</div>
                    </div>
                    <div class="metric-grid-sub">
                        {'🏰 Home Ground Advantage' if feat['team1_is_home'] else '✈️ Away Ground Record'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with v_col2:
            st.markdown(
                f"""
                <div class="metric-grid-card" style="margin-top: 0.6rem;">
                    <div>
                        <div class="metric-grid-label">{t2_info['short']} @ {venue_short}</div>
                        <div class="metric-grid-val">{feat['team2_venue_win_rate']*100:.1f}%</div>
                    </div>
                    <div class="metric-grid-sub">
                        {'🏰 Home Ground Advantage' if feat['team2_is_home'] else '✈️ Away Ground Record'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# TAB 2: Visual Deep-Dive (Plotly Charts)
# =============================================================================
with tab_analytics:
    st.markdown("### 📊 Interactive Matchup Telemetry")
    st.markdown(
        f"Deep statistical comparison between **{selected_t1}** and **{selected_t2}** at **{selected_venue}**."
    )

    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        st.markdown("##### 📈 Recent Form Momentum (5G vs 10G)")
        form_fig = create_form_comparison_chart(
            selected_t1,
            selected_t2,
            feat["team1_form_5"],
            feat["team2_form_5"],
            feat["team1_form_10"],
            feat["team2_form_10"],
        )
        st.plotly_chart(form_fig, use_container_width=True, config={"displayModeBar": False})

    with chart_c2:
        st.markdown(f"##### 🏟️ Ground Familiarity vs Career Win Rate")
        venue_fig = create_venue_comparison_chart(
            selected_t1,
            selected_t2,
            selected_venue,
            feat["team1_venue_win_rate"],
            feat["team2_venue_win_rate"],
            feat["team1_overall_win_rate"],
            feat["team2_overall_win_rate"],
        )
        st.plotly_chart(venue_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("##### ⚔️ Head-to-Head Encounter Breakdown")
    h2h_fig = create_h2h_chart(selected_t1, selected_t2, feat["h2h_matches"], feat["team1_h2h_win_rate"])
    st.plotly_chart(h2h_fig, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# TAB 3: Model Intelligence
# =============================================================================
with tab_model:
    st.markdown("### 🧠 Machine Learning Model Architecture")
    st.markdown(
        """
        The outcome forecasting engine employs a regularized **RandomForestClassifier** pipeline wrapped in scikit-learn's `ColumnTransformer`.
        To strictly prevent lookahead temporal leakage, the model was tuned and validated using **TimeSeriesSplit (5 folds)** across 17 seasons of historical data.
        """
    )

    m_col1, m_col2 = st.columns([1, 1.2])

    with m_col1:
        st.markdown("##### ⚙️ Tuned Hyperparameters")
        st.markdown(
            """
            | Hyperparameter | Value | Description |
            |---|:---:|---|
            | `n_estimators` | `150` | Number of decision trees in ensemble |
            | `max_depth` | `8` | Regularized depth to prevent noise memorization |
            | `min_samples_leaf` | `4` | Minimum samples required at terminal node |
            | `min_samples_split` | `5` | Minimum samples required to split internal node |
            | `max_features` | `'sqrt'` | Subsamples features to de-correlate trees |
            | `class_weight` | `None` | Balanced binary target (50.0% / 50.0%) |
            | `random_state` | `42` | Ensures deterministic, reproducible inference |
            """
        )

        st.markdown("##### 🎯 Benchmark Progression")
        st.markdown(
            """
            - **Majority Class Guess**: `45.75%`
            - **Career Win-Rate Heuristic**: `49.39%`
            - **Recent Form (5G) Baseline**: `50.61%`
            - **Toss Winner Baseline (Target to Beat)**: `51.42%`
            - **Untuned RandomForest (Overfit)**: `46.96%`
            - **Tuned RandomForest Pipeline**: **`51.42%`** *(+4.46% improvement)*
            """
        )

    with m_col2:
        st.markdown("##### 🏆 Top Gini Feature Importances")
        feat_importance_data = {
            "Feature": [
                "Venue Win Rate Differential",
                "Career Win Rate Differential",
                "Team 1 Career Win Rate",
                "Team 2 Career Win Rate",
                "Team 1 Venue Win Rate",
                "Team 2 Venue Win Rate",
                "H2H Matches Played",
                "Team 1 H2H Win Rate",
                "Team 1 Form (Last 5)",
                "Team 2 Form (Last 5)",
            ],
            "Importance": [8.43, 8.02, 7.65, 7.42, 6.54, 6.38, 5.98, 5.42, 4.89, 4.71],
        }
        df_feat = pd.DataFrame(feat_importance_data).sort_values("Importance", ascending=True)

        import plotly.express as px
        fig_feat = px.bar(
            df_feat,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_feat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=20, t=10, b=10),
            height=340,
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", ticksuffix="%"),
            yaxis=dict(tickfont=dict(color="#E2E8F0")),
        )
        fig_feat.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# TAB 4: About & Data Sources
# =============================================================================
with tab_about:
    st.markdown("### ℹ️ About the Project & Data Sources")
    st.markdown(
        """
        #### 📦 Dataset Provenance
        - **Source**: [Cricsheet.org](https://cricsheet.org/) ball-by-ball and match-summary dataset.
        - **Coverage**: Indian Premier League fixtures from 2008 through 2024 (17 seasons).
        - **Records**: 1,234 cleaned, non-abandoned competitive matches.
        - **License**: Open Data Commons Open Database License (ODbL).

        #### 🛡️ Leak-Free Engineering Methodology
        To guarantee zero temporal lookahead leakage, every predictive feature (rolling 5/10 match momentum, head-to-head records, venue familiarity win rates) is computed strictly using matches that concluded **before the match date**.

        #### 🌐 Open Source & Deployment
        - **GitHub Repository**: [sethubpathy/cricket-match-prediction](https://github.com/sethubpathy/cricket-match-prediction)
        - **Deployment Platform**: Streamlit Community Cloud
        - **Tech Stack**: Python 3.13, Pandas, NumPy, Scikit-Learn, Joblib, Plotly, Streamlit.
        """
    )


# -----------------------------------------------------------------------------
# 8. Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-container">
        <div>
            🏏 <b>Cricket Match Outcome Prediction System</b> • Built with Python, Scikit-Learn & Streamlit
        </div>
        <div style="margin-top: 0.35rem;">
            Source Code available on <a href="https://github.com/sethubpathy/cricket-match-prediction" target="_blank">GitHub (sethubpathy/cricket-match-prediction)</a> • Data powered by Cricsheet (ODbL)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
