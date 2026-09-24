"""Streamlit Web Application for Cricket Match Outcome Prediction.

A professional, product-grade sports-analytics dashboard predicting IPL match winners
and calibrated win probabilities using historical telemetry and a tuned RandomForestClassifier.
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
    get_icon,
    create_win_gauge,
    create_h2h_chart,
    create_form_comparison_chart,
    create_venue_comparison_chart,
    create_feature_importance_chart,
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
css_path = os.path.join(current_dir, "src", "styles.css")
if not os.path.exists(css_path):
    css_path = os.path.join(subfolder_src, "src", "styles.css")

if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        custom_css = f.read()
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


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
    """Callback to apply quick preset values into widget state."""
    st.session_state["sb_t1"] = team1
    st.session_state["sb_t2"] = team2
    st.session_state["sb_ven"] = venue
    st.session_state["sb_tw"] = toss_winner
    st.session_state["rb_td"] = toss_decision
    st.session_state["preset_status_msg"] = f"Loaded preset: **{name}**. Ready to predict."


def swap_teams_callback():
    """Swap Team 1 and Team 2, adjusting toss winner accordingly."""
    curr_t1 = st.session_state["sb_t1"]
    curr_t2 = st.session_state["sb_t2"]
    st.session_state["sb_t1"] = curr_t2
    st.session_state["sb_t2"] = curr_t1
    # If toss winner was team 1, keep it with team 1 (now current t2) or align
    if st.session_state["sb_tw"] == curr_t1:
        st.session_state["sb_tw"] = curr_t1
    elif st.session_state["sb_tw"] == curr_t2:
        st.session_state["sb_tw"] = curr_t2


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
# 5. Slim Top Navigation Bar (Linear / Vercel Aesthetic)
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <header class="top-nav-bar" role="banner">
        <div class="nav-brand">
            <span class="nav-logo-mark" aria-hidden="true">🏏</span>
            <div class="nav-titles">
                <h1 class="nav-title" style="margin: 0; padding: 0;">Cricket Match Outcome Predictor</h1>
                <span class="nav-subtitle">IPL AI Win Probability & Match Outcome Forecasting</span>
            </div>
        </div>
        <div class="nav-right">
            <div class="kpi-strip" role="region" aria-label="Model Specifications">
                <div class="kpi-tile">
                    <span class="kpi-label">Algorithm</span>
                    <span class="kpi-val">RandomForest</span>
                </div>
                <div class="kpi-tile">
                    <span class="kpi-label">Dataset</span>
                    <span class="kpi-val">17 Seasons</span>
                </div>
                <div class="kpi-tile">
                    <span class="kpi-label">Telemetry</span>
                    <span class="kpi-val">28 Features</span>
                </div>
                <div class="kpi-tile">
                    <span class="kpi-label">Validation</span>
                    <span class="kpi-val">Time-Aware</span>
                </div>
            </div>
            <a href="https://github.com/sethubpathy/cricket-match-prediction" target="_blank" class="nav-github-link" aria-label="View Source on GitHub">
                {get_icon("github", 15)}
                <span>GitHub</span>
            </a>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 6. Navigation Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_analytics, tab_model, tab_about = st.tabs([
    "Match Predictor",
    "Matchup Deep-Dive",
    "Model Intelligence",
    "Data & Methodology",
])


# =============================================================================
# TAB 1: Core Match Predictor (Sticky Left Setup + Right Results)
# =============================================================================
with tab_predict:
    col_input, col_results = st.columns([1.0, 2.1], gap="large")

    # ------------------ LEFT COLUMN: MATCH SETUP ------------------
    with col_input:
        st.markdown(
            f"""
            <div class="setup-header">
                <div class="setup-title">
                    {get_icon("sliders", 16, "#3B82F6")}
                    <span>Match Setup</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Match Presets Chips
        st.markdown(
            f"""
            <div class="presets-label">
                {get_icon("sparkles", 13, "#3B82F6")}
                <span>Quick Match Presets</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1_active = (st.session_state.get("sb_t1") == "Chennai Super Kings" and st.session_state.get("sb_t2") == "Mumbai Indians")
        p2_active = (st.session_state.get("sb_t1") == "Kolkata Knight Riders" and st.session_state.get("sb_t2") == "Royal Challengers Bengaluru")
        p3_active = (st.session_state.get("sb_t1") == "Gujarat Titans" and st.session_state.get("sb_t2") == "Rajasthan Royals")

        preset_c1, preset_c2, preset_c3 = st.columns(3)

        with preset_c1:
            st.button(
                "MI vs CSK",
                help="El Clásico at Wankhede Stadium",
                use_container_width=True,
                key="btn_preset_1",
                on_click=apply_preset,
                args=("Chennai Super Kings", "Mumbai Indians", "Wankhede Stadium", "Chennai Super Kings", "field", "MI vs CSK"),
            )

        with preset_c2:
            st.button(
                "KKR vs RCB",
                help="Royal Derby at Eden Gardens",
                use_container_width=True,
                key="btn_preset_2",
                on_click=apply_preset,
                args=("Kolkata Knight Riders", "Royal Challengers Bengaluru", "Eden Gardens", "Kolkata Knight Riders", "field", "KKR vs RCB"),
            )

        with preset_c3:
            st.button(
                "GT vs RR",
                help="Finalist Rematch at Ahmedabad",
                use_container_width=True,
                key="btn_preset_3",
                on_click=apply_preset,
                args=("Gujarat Titans", "Rajasthan Royals", "Narendra Modi Stadium, Ahmedabad", "Gujarat Titans", "field", "GT vs RR"),
            )

        if st.session_state.get("preset_status_msg"):
            st.caption(f"⚡ {st.session_state['preset_status_msg']}")

        st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)

        # Team 1 Selection
        selected_t1 = st.selectbox(
            "Select Team 1",
            options=teams_list,
            key="sb_t1",
            on_change=on_team1_change,
        )

        # Swap Teams Action Button
        swap_col1, swap_col2, swap_col3 = st.columns([1, 2, 1])
        with swap_col2:
            st.markdown('<div class="btn-swap-teams">', unsafe_allow_html=True)
            st.button(
                "⇄ Swap Teams",
                key="btn_swap_teams",
                help="Swap Team 1 and Team 2 positions",
                use_container_width=True,
                on_click=swap_teams_callback,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Team 2 Selection (strictly validated to prevent duplicate team)
        available_t2 = [t for t in teams_list if t != selected_t1]
        if st.session_state.get("sb_t2") not in available_t2:
            st.session_state["sb_t2"] = available_t2[0]

        selected_t2 = st.selectbox(
            "Select Team 2",
            options=available_t2,
            key="sb_t2",
            on_change=on_team2_change,
        )

        # Venue Selection
        selected_venue = st.selectbox(
            "Match Venue",
            options=venues_list,
            key="sb_ven",
        )

        # Toss Winner Selection (strictly candidate teams)
        toss_candidates = [selected_t1, selected_t2]
        if st.session_state.get("sb_tw") not in toss_candidates:
            st.session_state["sb_tw"] = toss_candidates[0]

        selected_toss_winner = st.selectbox(
            "Toss Winner",
            options=toss_candidates,
            key="sb_tw",
        )

        # Toss Decision Segmented Control
        selected_toss_decision = st.radio(
            "Toss Decision",
            options=["field", "bat"],
            key="rb_td",
            format_func=lambda x: "Field (Chasing)" if x == "field" else "Bat (Defending)",
            horizontal=True,
        )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        predict_btn = st.button(
            "🔮 Predict Match Outcome",
            type="primary",
            use_container_width=True,
            key="btn_predict",
        )

    # ------------------ RIGHT COLUMN: RESULTS & TELEMETRY ------------------
    with col_results:
        current_params = (selected_t1, selected_t2, selected_venue, selected_toss_winner, selected_toss_decision)

        # Compute or load prediction
        if "prediction_result" not in st.session_state or predict_btn:
            with st.spinner("Analyzing matchup telemetry and computing calibrated probability..."):
                st.session_state["prediction_result"] = predict_match(
                    team1=selected_t1,
                    team2=selected_t2,
                    venue=selected_venue,
                    toss_winner=selected_toss_winner,
                    toss_decision=selected_toss_decision,
                )
                st.session_state["prediction_params"] = current_params

        res = st.session_state.get("prediction_result")

        if not res:
            # Meaningful empty state before first prediction
            st.markdown(
                f"""
                <div class="empty-state-card">
                    <div class="empty-state-icon">{get_icon("brain", 48, "#3B82F6")}</div>
                    <div class="empty-state-title">Ready for Match Simulation</div>
                    <div class="empty-state-desc">
                        Select two competing IPL franchises, match venue, and toss outcome on the left panel,
                        then click <strong>'Predict Match Outcome'</strong> to generate statistical win probabilities.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            saved_params = st.session_state.get("prediction_params")
            if saved_params and saved_params != current_params:
                st.warning("⚠️ Parameters updated. Click '🔮 Predict Match Outcome' to refresh forecast.")

            winner = res["predicted_winner"]
            win_prob = res["win_probability"]
            t1_prob = res["team1_win_probability"]
            t2_prob = res["team2_win_probability"]
            feat = res["features"]

            # Team metadata
            t1_info = get_team_info(selected_t1)
            t2_info = get_team_info(selected_t2)
            winner_info = get_team_info(winner)
            rival_name = selected_t2 if winner == selected_t1 else selected_t1
            rival_prob = t2_prob if winner == selected_t1 else t1_prob
            rival_info = get_team_info(rival_name)

            # AI Tactical Interpretation
            conf_label, badge_style, tactical_bullets, disclaimer_note = generate_ai_summary(res)

            # Franchise Accent Color & Card Background
            win_color = winner_info["primary"]
            card_gradient = f"linear-gradient(180deg, {win_color}18 0%, rgba(17, 24, 39, 0.95) 75%)"

            # Results Wrapper with Fade-in CSS
            st.markdown('<div class="results-container">', unsafe_allow_html=True)

            # 1. Hero Winner Card
            st.markdown(
                f"""
                <div class="hero-winner-card" style="background: {card_gradient}; border: 1px solid rgba(255, 255, 255, 0.08); border-top: 4px solid {win_color};">
                    <div class="winner-meta-row">
                        <span class="winner-tag" style="background: rgba(255, 255, 255, 0.06); color: #F3F4F6; border: 1px solid rgba(255, 255, 255, 0.1);">
                            PROJECTED MATCH WINNER
                        </span>
                        <span class="confidence-pill" style="{badge_style}">
                            {conf_label}
                        </span>
                    </div>
                    <div class="winner-main-row">
                        <div class="winner-name-group">
                            <span class="winner-name-text">{winner}</span>
                        </div>
                        <div class="winner-prob-group">
                            <div class="winner-prob-stat stat-num" style="color: #FFFFFF;">{win_prob:.1f}%</div>
                            <div class="winner-prob-sub">Win Probability</div>
                        </div>
                    </div>
                    <div class="winner-vs-line">
                        <span>Predicted Edge: <strong>+{abs(t1_prob - t2_prob):.1f}%</strong> over opponent</span>
                        <span>vs. <strong>{rival_name}</strong> ({rival_prob:.1f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Head-to-Head Split Probability Bar
            st.markdown(
                f"""
                <div class="h2h-split-container">
                    <div class="h2h-labels-row">
                        <div class="h2h-team-chip">
                            <span class="team-color-dot" style="background-color: {t1_info['primary']};"></span>
                            <span style="color: #F3F4F6; font-weight: 700;">{selected_t1}</span>
                            <span class="stat-num" style="color: #9CA3AF; margin-left: 4px;">({t1_prob:.1f}%)</span>
                        </div>
                        <div class="h2h-team-chip">
                            <span class="stat-num" style="color: #9CA3AF; margin-right: 4px;">({t2_prob:.1f}%)</span>
                            <span style="color: #F3F4F6; font-weight: 700;">{selected_t2}</span>
                            <span class="team-color-dot" style="background-color: {t2_info['primary']};"></span>
                        </div>
                    </div>
                    <div class="h2h-bar-track">
                        <div class="h2h-bar-segment-t1" style="width: {t1_prob}%; background-color: {t1_info['primary']};"></div>
                        <div class="h2h-bar-segment-t2" style="width: {t2_prob}%; background-color: {t2_info['primary']};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 3. Gauge & Tactical Model Analysis
            g_col, a_col = st.columns([1, 1.25], gap="medium")

            with g_col:
                gauge_fig = create_win_gauge(winner, win_prob, selected_t1, selected_t2, t1_prob, t2_prob)
                st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

            with a_col:
                bullets_html = "".join([
                    f'<li class="tactical-bullet-item"><span class="bullet-icon">{get_icon("check", 14, "#3B82F6")}</span><div>{bullet}</div></li>'
                    for bullet in tactical_bullets
                ])
                st.markdown(
                    f"""
                    <div class="tactical-analysis-card">
                        <div>
                            <div class="tactical-header">
                                {get_icon("brain", 15, "#3B82F6")}
                                <span>Tactical Model Analysis</span>
                            </div>
                            <ul class="tactical-bullets">
                                {bullets_html}
                            </ul>
                        </div>
                        <div class="tactical-disclaimer">
                            ℹ️ {disclaimer_note}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 4. Contextual Match Intelligence (4 Equal-Height Responsive Metric Cards)
            st.markdown(
                f"""
                <div class="metrics-section-title">
                    {get_icon("activity", 15, "#3B82F6")}
                    <span>Matchup Context & Historical Records</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Get actual last-5 results for W/L badges
            t1_results_5 = get_team_recent_results(matches_df, selected_t1, 5)
            t2_results_5 = get_team_recent_results(matches_df, selected_t2, 5)

            def render_wl_pills(results: List[str]) -> str:
                pills = "".join([f'<span class="wl-pill {"w" if r == "W" else "l"}">{r}</span>' for r in results])
                return f'<div class="wl-row">{pills}</div>'

            venue_short = selected_venue.split(",")[0]

            st.markdown(
                f"""
                <div class="metrics-responsive-grid">
                    <!-- Card 1: Head-to-Head -->
                    <div class="metric-card">
                        <div>
                            <div class="metric-card-top">
                                <span class="metric-card-label">Head-to-Head</span>
                                <span class="metric-card-icon">{get_icon("swords", 15)}</span>
                            </div>
                            <div class="metric-card-val stat-num">{feat['h2h_matches']} Games</div>
                        </div>
                        <div class="metric-card-context">
                            <strong>{t1_info['short']}</strong>: {feat['team1_h2h_win_rate']*100:.0f}% win rate<br>
                            <strong>{t2_info['short']}</strong>: {(1.0 - feat['team1_h2h_win_rate'])*100:.0f}% win rate
                        </div>
                    </div>

                    <!-- Card 2: Team 1 Form -->
                    <div class="metric-card">
                        <div>
                            <div class="metric-card-top">
                                <span class="metric-card-label">{t1_info['short']} Form</span>
                                <span class="metric-card-icon">{get_icon("activity", 15)}</span>
                            </div>
                            <div class="metric-card-val stat-num">{feat['team1_form_5']*100:.0f}%</div>
                            {render_wl_pills(t1_results_5)}
                        </div>
                        <div class="metric-card-context">
                            Last 10 Games: <strong>{feat['team1_form_10']*100:.0f}%</strong> win rate
                        </div>
                    </div>

                    <!-- Card 3: Team 2 Form -->
                    <div class="metric-card">
                        <div>
                            <div class="metric-card-top">
                                <span class="metric-card-label">{t2_info['short']} Form</span>
                                <span class="metric-card-icon">{get_icon("activity", 15)}</span>
                            </div>
                            <div class="metric-card-val stat-num">{feat['team2_form_5']*100:.0f}%</div>
                            {render_wl_pills(t2_results_5)}
                        </div>
                        <div class="metric-card-context">
                            Last 10 Games: <strong>{feat['team2_form_10']*100:.0f}%</strong> win rate
                        </div>
                    </div>

                    <!-- Card 4: Venue History -->
                    <div class="metric-card">
                        <div>
                            <div class="metric-card-top">
                                <span class="metric-card-label">Ground Record</span>
                                <span class="metric-card-icon">{get_icon("map-pin", 15)}</span>
                            </div>
                            <div class="metric-card-val stat-num">{feat['team1_venue_win_rate']*100:.0f}% / {feat['team2_venue_win_rate']*100:.0f}%</div>
                        </div>
                        <div class="metric-card-context">
                            {venue_short}<br>
                            {t1_info['short']} ({'Home' if feat['team1_is_home'] else 'Away'}) vs. {t2_info['short']} ({'Home' if feat['team2_is_home'] else 'Away'})
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Share / Copy Forecast Option
            with st.expander("📋 Copy Forecast Summary / Share Prediction"):
                summary_text = (
                    f"IPL Match Forecast:\n"
                    f"Predicted Winner: {winner} ({win_prob:.1f}% Probability)\n"
                    f"Matchup: {selected_t1} ({t1_prob:.1f}%) vs. {selected_t2} ({t2_prob:.1f}%)\n"
                    f"Venue: {selected_venue}\n"
                    f"Toss: {selected_toss_winner} elected to {selected_toss_decision.upper()}\n"
                    f"Confidence: {conf_label}\n"
                    f"Source: Cricket Match Outcome Predictor (sethubpathy/cricket-match-prediction)"
                )
                st.code(summary_text, language="markdown")

            st.markdown('</div>', unsafe_allow_html=True)  # Close results-container


# =============================================================================
# TAB 2: Matchup Deep-Dive (Plotly Telemetry)
# =============================================================================
with tab_analytics:
    st.markdown("### Matchup Deep-Dive & Telemetry")
    st.markdown(
        f"Granular comparison between **{selected_t1}** and **{selected_t2}** at **{selected_venue}**."
    )

    chart_c1, chart_c2 = st.columns(2, gap="medium")

    with chart_c1:
        st.markdown("##### Recent Form Momentum (5G vs 10G)")
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
        st.markdown("##### Ground Familiarity vs Career Win Rate")
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

    st.markdown("##### Head-to-Head Encounter Breakdown")
    h2h_fig = create_h2h_chart(selected_t1, selected_t2, feat["h2h_matches"], feat["team1_h2h_win_rate"])
    st.plotly_chart(h2h_fig, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# TAB 3: Model Intelligence
# =============================================================================
with tab_model:
    st.markdown("### Machine Learning Model Intelligence")
    st.markdown(
        """
        The outcome forecasting engine employs a regularized **RandomForestClassifier** pipeline wrapped in scikit-learn's `ColumnTransformer`.
        To strictly prevent lookahead temporal leakage, the model was validated using **TimeSeriesSplit (5 folds)** across 17 seasons of historical data.
        """
    )

    m_col1, m_col2 = st.columns([1, 1.25], gap="large")

    with m_col1:
        st.markdown("##### Tuned Hyperparameters")
        st.markdown(
            """
            | Hyperparameter | Value | Description |
            |---|:---:|---|
            | `n_estimators` | `150` | Number of decision trees in ensemble |
            | `max_depth` | `8` | Regularized depth to prevent memorization |
            | `min_samples_leaf` | `4` | Minimum samples required at terminal leaf node |
            | `min_samples_split` | `5` | Minimum samples required to split internal node |
            | `max_features` | `'sqrt'` | Subsamples features to de-correlate individual trees |
            | `class_weight` | `None` | Balanced binary target distribution (50.0% / 50.0%) |
            | `random_state` | `42` | Guarantees deterministic, reproducible inference |
            """
        )

        st.markdown("##### Benchmark Progression")
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
        st.markdown("##### Top Gini Feature Importances")
        feat_chart = create_feature_importance_chart()
        st.plotly_chart(feat_chart, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# TAB 4: About & Data Sources
# =============================================================================
with tab_about:
    st.markdown("### Data & Methodology")
    st.markdown(
        """
        #### Dataset Provenance
        - **Source**: [Cricsheet.org](https://cricsheet.org/) ball-by-ball and match-summary dataset.
        - **Coverage**: Indian Premier League fixtures from 2008 through 2024 (17 seasons).
        - **Records**: 1,234 cleaned, non-abandoned competitive matches.
        - **License**: Open Data Commons Open Database License (ODbL).

        #### Leak-Free Feature Engineering
        To guarantee zero temporal lookahead leakage, every predictive feature (rolling 5/10 match momentum, head-to-head records, venue familiarity win rates) is computed strictly using matches that concluded **before the match date**.

        #### Open Source & Deployment
        - **GitHub Repository**: [sethubpathy/cricket-match-prediction](https://github.com/sethubpathy/cricket-match-prediction)
        - **Deployment Platform**: Streamlit Community Cloud
        - **Tech Stack**: Python 3.13, Pandas, NumPy, Scikit-Learn, Joblib, Plotly, Streamlit.
        """
    )


# -----------------------------------------------------------------------------
# 7. Minimal Disclaimer Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <footer class="app-footer" role="contentinfo">
        <div>
            🏏 <strong>Cricket Match Outcome Predictor</strong> • High-Performance IPL Analytics Engine
        </div>
        <div>
            Predictions are statistical estimates for educational and analytical purposes • Data powered by Cricsheet (ODbL)
        </div>
        <div style="margin-top: 0.25rem;">
            Source code available on <a href="https://github.com/sethubpathy/cricket-match-prediction" target="_blank">GitHub</a>
        </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
