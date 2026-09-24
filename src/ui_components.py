"""UI helper functions, charts, and styling components for Cricket Match Predictor."""

from typing import Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Official franchise branding colors and metadata
TEAM_METADATA: Dict[str, Dict[str, str]] = {
    "Chennai Super Kings": {
        "primary": "#FACC15",     # CSK Canary Yellow
        "secondary": "#1E3A8A",   # Lion Navy Blue
        "text_on_primary": "#000000",
        "short": "CSK",
        "emoji": "🦁",
    },
    "Mumbai Indians": {
        "primary": "#004BA0",     # MI Royal Blue
        "secondary": "#D4AF37",   # Gold
        "text_on_primary": "#FFFFFF",
        "short": "MI",
        "emoji": "🌀",
    },
    "Kolkata Knight Riders": {
        "primary": "#3A225D",     # KKR Knight Purple
        "secondary": "#F59E0B",   # Gold
        "text_on_primary": "#FFFFFF",
        "short": "KKR",
        "emoji": "⚔️",
    },
    "Royal Challengers Bengaluru": {
        "primary": "#DC2626",     # RCB Crimson Red
        "secondary": "#1F2937",   # Charcoal Black / Gold
        "text_on_primary": "#FFFFFF",
        "short": "RCB",
        "emoji": "👑",
    },
    "Gujarat Titans": {
        "primary": "#1E293B",     # Deep Slate Navy
        "secondary": "#0D9488",   # Titans Teal
        "text_on_primary": "#FFFFFF",
        "short": "GT",
        "emoji": "🛡️",
    },
    "Rajasthan Royals": {
        "primary": "#EC4899",     # Royals Pink
        "secondary": "#1E3A8A",   # Royal Navy
        "text_on_primary": "#FFFFFF",
        "short": "RR",
        "emoji": "👑",
    },
    "Delhi Capitals": {
        "primary": "#0284C7",     # DC Blue
        "secondary": "#EF4444",   # DC Red
        "text_on_primary": "#FFFFFF",
        "short": "DC",
        "emoji": "🐯",
    },
    "Punjab Kings": {
        "primary": "#E11D48",     # Crimson Red
        "secondary": "#FBBF24",   # Gold
        "text_on_primary": "#FFFFFF",
        "short": "PBKS",
        "emoji": "🦁",
    },
    "Sunrisers Hyderabad": {
        "primary": "#EA580C",     # SRH Orange
        "secondary": "#18181B",   # Black
        "text_on_primary": "#FFFFFF",
        "short": "SRH",
        "emoji": "🦅",
    },
    "Lucknow Super Giants": {
        "primary": "#06B6D4",     # LSG Sky Blue
        "secondary": "#F97316",   # Orange
        "text_on_primary": "#FFFFFF",
        "short": "LSG",
        "emoji": "⚡",
    },
    "Rising Pune Supergiant": {
        "primary": "#BE185D",
        "secondary": "#581C87",
        "text_on_primary": "#FFFFFF",
        "short": "RPSG",
        "emoji": "🌟",
    },
    "Deccan Chargers": {
        "primary": "#3B82F6",
        "secondary": "#D97706",
        "text_on_primary": "#FFFFFF",
        "short": "DCG",
        "emoji": "⚡",
    },
    "Pune Warriors": {
        "primary": "#334155",
        "secondary": "#15803D",
        "text_on_primary": "#FFFFFF",
        "short": "PWI",
        "emoji": "⚔️",
    },
    "Gujarat Lions": {
        "primary": "#C2410C",
        "secondary": "#78350F",
        "text_on_primary": "#FFFFFF",
        "short": "GL",
        "emoji": "🦁",
    },
    "Kochi Tuskers Kerala": {
        "primary": "#7C3AED",
        "secondary": "#DB2777",
        "text_on_primary": "#FFFFFF",
        "short": "KTK",
        "emoji": "🐘",
    },
}

# Fallback palette for unknown/custom teams
DEFAULT_TEAM_META = {
    "primary": "#3B82F6",
    "secondary": "#10B981",
    "text_on_primary": "#FFFFFF",
    "short": "TEAM",
    "emoji": "🏏",
}


def get_team_info(team_name: str) -> Dict[str, str]:
    """Retrieve color and branding metadata for a franchise."""
    return TEAM_METADATA.get(team_name, DEFAULT_TEAM_META)


def get_team_recent_results(df: pd.DataFrame, team: str, n: int = 5) -> List[str]:
    """Extract chronological W/L sequence for a team over its last n fixtures."""
    sub = df[(df["team1"] == team) | (df["team2"] == team)].sort_values("date")
    recent = sub.tail(n)
    return ["W" if row["winner"] == team else "L" for _, row in recent.iterrows()]


def create_win_gauge(
    winner_name: str,
    win_probability: float,
    team1_name: str,
    team2_name: str,
    t1_prob: float,
    t2_prob: float,
) -> go.Figure:
    """Create a sleek, modern circular gauge / donut chart displaying win probability."""
    win_color = get_team_info(winner_name)["primary"]
    # If winner is team1, use t1_prob; otherwise t2_prob
    prob_val = win_probability

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_val,
            number={"suffix": "%", "font": {"size": 42, "color": "#F8FAFC", "family": "Inter, sans-serif"}},
            title={
                "text": f"<b>{get_team_info(winner_name)['short']} WIN PROBABILITY</b>",
                "font": {"size": 13, "color": "#94A3B8", "family": "Inter, sans-serif"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#475569",
                    "tickfont": {"color": "#64748B", "size": 10},
                    "tickvals": [0, 25, 50, 75, 100],
                },
                "bar": {"color": win_color, "thickness": 0.45},
                "bgcolor": "rgba(255, 255, 255, 0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(255, 255, 255, 0.03)"},
                    {"range": [50, 100], "color": "rgba(255, 255, 255, 0.08)"},
                ],
                "threshold": {
                    "line": {"color": "#38BDF8", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=45, b=25),
        height=220,
        font=dict(color="#F8FAFC", family="Inter, sans-serif"),
    )
    return fig


def create_h2h_chart(
    team1: str,
    team2: str,
    h2h_matches: int,
    t1_win_rate: float,
) -> go.Figure:
    """Create a Plotly horizontal comparison bar chart of head-to-head records."""
    t1_meta = get_team_info(team1)
    t2_meta = get_team_info(team2)

    t1_wins = int(round(h2h_matches * t1_win_rate))
    t2_wins = h2h_matches - t1_wins

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=["Head-to-Head"],
            x=[t1_wins],
            name=f"{t1_meta['short']} ({t1_wins} Wins)",
            orientation="h",
            marker=dict(color=t1_meta["primary"], line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=f"{t1_meta['short']}: {t1_wins} ({t1_win_rate*100:.0f}%)",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", size=12, family="Inter"),
            hoverinfo="name+x",
        )
    )

    fig.add_trace(
        go.Bar(
            y=["Head-to-Head"],
            x=[t2_wins],
            name=f"{t2_meta['short']} ({t2_wins} Wins)",
            orientation="h",
            marker=dict(color=t2_meta["primary"], line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=f"{t2_meta['short']}: {t2_wins} ({(1 - t1_win_rate)*100:.0f}%)",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", size=12, family="Inter"),
            hoverinfo="name+x",
        )
    )

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=90,
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )
    return fig


def create_form_comparison_chart(
    team1: str,
    team2: str,
    t1_form_5: float,
    t2_form_5: float,
    t1_form_10: float,
    t2_form_10: float,
) -> go.Figure:
    """Create a grouped bar chart comparing recent form (last 5 & 10 matches)."""
    t1_meta = get_team_info(team1)
    t2_meta = get_team_info(team2)

    categories = ["Last 5 Games", "Last 10 Games"]

    fig = go.Figure(
        data=[
            go.Bar(
                name=t1_meta["short"],
                x=categories,
                y=[t1_form_5 * 100, t1_form_10 * 100],
                marker=dict(color=t1_meta["primary"], line=dict(color="rgba(255,255,255,0.2)", width=1)),
                text=[f"{t1_form_5*100:.0f}%", f"{t1_form_10*100:.0f}%"],
                textposition="outside",
                textfont=dict(color="#CBD5E1", size=11),
            ),
            go.Bar(
                name=t2_meta["short"],
                x=categories,
                y=[t2_form_5 * 100, t2_form_10 * 100],
                marker=dict(color=t2_meta["primary"], line=dict(color="rgba(255,255,255,0.2)", width=1)),
                text=[f"{t2_form_5*100:.0f}%", f"{t2_form_10*100:.0f}%"],
                textposition="outside",
                textfont=dict(color="#CBD5E1", size=11),
            ),
        ]
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=240,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#E2E8F0", size=11),
        ),
        yaxis=dict(
            range=[0, 115],
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94A3B8"),
            ticksuffix="%",
        ),
        xaxis=dict(
            tickfont=dict(color="#E2E8F0", size=12),
            showgrid=False,
        ),
    )
    return fig


def create_venue_comparison_chart(
    team1: str,
    team2: str,
    venue: str,
    t1_venue_wr: float,
    t2_venue_wr: float,
    t1_overall_wr: float,
    t2_overall_wr: float,
) -> go.Figure:
    """Compare team win rates at match venue against overall career win rate."""
    t1_meta = get_team_info(team1)
    t2_meta = get_team_info(team2)

    categories = ["At Match Venue", "Career Overall"]

    fig = go.Figure(
        data=[
            go.Bar(
                name=t1_meta["short"],
                x=categories,
                y=[t1_venue_wr * 100, t1_overall_wr * 100],
                marker=dict(color=t1_meta["primary"], line=dict(color="rgba(255,255,255,0.2)", width=1)),
                text=[f"{t1_venue_wr*100:.1f}%", f"{t1_overall_wr*100:.1f}%"],
                textposition="outside",
                textfont=dict(color="#CBD5E1", size=11),
            ),
            go.Bar(
                name=t2_meta["short"],
                x=categories,
                y=[t2_venue_wr * 100, t2_overall_wr * 100],
                marker=dict(color=t2_meta["primary"], line=dict(color="rgba(255,255,255,0.2)", width=1)),
                text=[f"{t2_venue_wr*100:.1f}%", f"{t2_overall_wr*100:.1f}%"],
                textposition="outside",
                textfont=dict(color="#CBD5E1", size=11),
            ),
        ]
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=240,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#E2E8F0", size=11),
        ),
        yaxis=dict(
            range=[0, 115],
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94A3B8"),
            ticksuffix="%",
        ),
        xaxis=dict(
            tickfont=dict(color="#E2E8F0", size=12),
            showgrid=False,
        ),
    )
    return fig


def generate_ai_summary(res: Dict) -> Tuple[str, str, str]:
    """Generate domain interpretation explaining why model predicts outcome.
    
    Returns:
        (confidence_label, badge_color, explanation_text)
    """
    winner = res["predicted_winner"]
    win_prob = res["win_probability"]
    team1 = res["team1"]
    team2 = res["team2"]
    venue = res["venue"].split(",")[0]
    feat = res["features"]

    # Confidence classification
    margin = abs(res["team1_win_probability"] - res["team2_win_probability"])
    if margin >= 16.0:
        conf_label = "HIGH CONFIDENCE"
        badge_style = "background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4);"
    elif margin >= 6.0:
        conf_label = "MODERATE ADVANTAGE"
        badge_style = "background: rgba(56, 189, 248, 0.2); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4);"
    else:
        conf_label = "TIGHT CONTEST"
        badge_style = "background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.4);"

    is_winner_t1 = (winner == team1)
    winner_form = feat["team1_form_5"] if is_winner_t1 else feat["team2_form_5"]
    rival_form = feat["team2_form_5"] if is_winner_t1 else feat["team1_form_5"]
    winner_vwr = feat["team1_venue_win_rate"] if is_winner_t1 else feat["team2_venue_win_rate"]
    rival_vwr = feat["team2_venue_win_rate"] if is_winner_t1 else feat["team1_venue_win_rate"]
    winner_is_home = feat["team1_is_home"] if is_winner_t1 else feat["team2_is_home"]

    factors = []
    # Factor 1: Venue dominance
    if winner_vwr > rival_vwr + 0.05:
        factors.append(f"significant ground mastery at <b>{venue}</b> ({winner_vwr*100:.1f}% win rate vs. {rival_vwr*100:.1f}%)")
    elif winner_is_home:
        factors.append(f"home ground familiarity at <b>{venue}</b>")

    # Factor 2: Form momentum
    if winner_form > rival_form:
        factors.append(f"superior recent momentum ({winner_form*100:.0f}% in last 5 games vs. {rival_form*100:.0f}%)")
    elif winner_form >= 0.6:
        factors.append(f"solid recent form ({winner_form*100:.0f}% in last 5 fixtures)")

    # Factor 3: Head-to-Head
    h2h_matches = feat["h2h_matches"]
    t1_h2h = feat["team1_h2h_win_rate"]
    winner_h2h = t1_h2h if is_winner_t1 else (1.0 - t1_h2h)
    if h2h_matches >= 5 and winner_h2h > 0.52:
        factors.append(f"favorable head-to-head historical edge ({winner_h2h*100:.1f}% across {h2h_matches} matches)")

    # Factor 4: Toss decision
    if res["toss_winner"] == winner:
        factors.append(f"tactical advantage of winning the toss and electing to <b>{res['toss_decision'].upper()}</b>")

    if not factors:
        factors.append("balanced franchise career win metrics and calibrated probabilistic ensemble features")

    reasons_str = "; ".join(factors)
    summary_text = (
        f"The tuned RandomForest model projects <b>{winner}</b> to have the upper hand with a "
        f"<b>{win_prob:.1f}%</b> win probability. Key statistical catalysts include {reasons_str}."
    )

    return conf_label, badge_style, summary_text
