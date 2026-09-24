"""UI helper functions, charts, and styling components for Cricket Match Predictor.

Product-grade sports-analytics UI components designed with Linear/Vercel styling,
ESPNcricinfo data density, and WCAG AA contrast compliance.
"""

from typing import Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# Official Franchise Branding & Colors (User Specified Hex Tokens)
# -----------------------------------------------------------------------------
TEAM_METADATA: Dict[str, Dict[str, str]] = {
    "Chennai Super Kings": {
        "primary": "#F9CD05",     # CSK Canary Yellow
        "secondary": "#1E3A8A",   # Lion Navy Blue
        "text_on_primary": "#000000",
        "short": "CSK",
    },
    "Mumbai Indians": {
        "primary": "#004BA0",     # MI Royal Blue
        "secondary": "#0A2540",   # Deep Navy
        "text_on_primary": "#FFFFFF",
        "short": "MI",
    },
    "Kolkata Knight Riders": {
        "primary": "#3A225D",     # KKR Knight Purple
        "secondary": "#D97706",   # Gold
        "text_on_primary": "#FFFFFF",
        "short": "KKR",
    },
    "Royal Challengers Bengaluru": {
        "primary": "#D1171B",     # RCB Crimson Red
        "secondary": "#111827",   # Dark Charcoal
        "text_on_primary": "#FFFFFF",
        "short": "RCB",
    },
    "Gujarat Titans": {
        "primary": "#1B2133",     # GT Slate Navy
        "secondary": "#0D9488",   # Titans Teal
        "text_on_primary": "#FFFFFF",
        "short": "GT",
    },
    "Rajasthan Royals": {
        "primary": "#EA1A85",     # Royals Pink
        "secondary": "#1E3A8A",   # Royal Navy
        "text_on_primary": "#FFFFFF",
        "short": "RR",
    },
    "Delhi Capitals": {
        "primary": "#17449B",     # DC Blue
        "secondary": "#EF4444",   # DC Red
        "text_on_primary": "#FFFFFF",
        "short": "DC",
    },
    "Punjab Kings": {
        "primary": "#DD1F2D",     # PBKS Red
        "secondary": "#FBBF24",   # Gold
        "text_on_primary": "#FFFFFF",
        "short": "PBKS",
    },
    "Sunrisers Hyderabad": {
        "primary": "#FF822A",     # SRH Orange
        "secondary": "#18181B",   # Black
        "text_on_primary": "#FFFFFF",
        "short": "SRH",
    },
    "Lucknow Super Giants": {
        "primary": "#00A6E0",     # LSG Sky Blue
        "secondary": "#F97316",   # Orange
        "text_on_primary": "#FFFFFF",
        "short": "LSG",
    },
    "Rising Pune Supergiant": {
        "primary": "#BE185D",
        "secondary": "#581C87",
        "text_on_primary": "#FFFFFF",
        "short": "RPSG",
    },
    "Deccan Chargers": {
        "primary": "#2563EB",
        "secondary": "#D97706",
        "text_on_primary": "#FFFFFF",
        "short": "DCG",
    },
    "Pune Warriors": {
        "primary": "#334155",
        "secondary": "#15803D",
        "text_on_primary": "#FFFFFF",
        "short": "PWI",
    },
    "Gujarat Lions": {
        "primary": "#EA580C",
        "secondary": "#78350F",
        "text_on_primary": "#FFFFFF",
        "short": "GL",
    },
    "Kochi Tuskers Kerala": {
        "primary": "#7C3AED",
        "secondary": "#DB2777",
        "text_on_primary": "#FFFFFF",
        "short": "KTK",
    },
}

DEFAULT_TEAM_META = {
    "primary": "#3B82F6",
    "secondary": "#10B981",
    "text_on_primary": "#FFFFFF",
    "short": "TEAM",
}


def get_team_info(team_name: str) -> Dict[str, str]:
    """Retrieve color and branding metadata for a franchise."""
    return TEAM_METADATA.get(team_name, DEFAULT_TEAM_META)


def get_team_recent_results(df: pd.DataFrame, team: str, n: int = 5) -> List[str]:
    """Extract chronological W/L sequence for a team over its last n fixtures."""
    sub = df[(df["team1"] == team) | (df["team2"] == team)].sort_values("date")
    recent = sub.tail(n)
    return ["W" if row["winner"] == team else "L" for _, row in recent.iterrows()]


# -----------------------------------------------------------------------------
# SVG Icons Helper (Lucide Set, No Decorative Emojis)
# -----------------------------------------------------------------------------
SVG_ICONS = {
    "trophy": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path><path d="M4 22h16"></path><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"></path><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"></path><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"></path></svg>',
    "swords": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="14.5 17.5 3 6 3 3 6 3 17.5 14.5"></polyline><line x1="13" y1="19" x2="19" y2="13"></line><line x1="16" y1="16" x2="20" y2="20"></line><line x1="19" y1="21" x2="21" y2="19"></line><polyline points="14.5 6.5 18 3 21 3 21 6 17.5 9.5"></polyline><line x1="5" y1="14" x2="9" y2="18"></line><line x1="7" y1="17" x2="4" y2="20"></line><line x1="3" y1="19" x2="5" y2="21"></line></svg>',
    "activity": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    "map-pin": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"></path><circle cx="12" cy="10" r="3"></circle></svg>',
    "brain": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"></path><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"></path><path d="M12 5v13"></path></svg>',
    "sliders": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>',
    "refresh": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path><path d="M21 3v5h-5"></path><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path><path d="M8 16H3v5"></path></svg>',
    "info": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
    "github": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"></path></svg>',
    "check": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    "copy": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg>',
    "sparkles": '<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path></svg>',
}


def get_icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    """Return inline SVG string with requested size and color."""
    template = SVG_ICONS.get(name, "")
    return template.format(size=size, color=color)


# -----------------------------------------------------------------------------
# 1. Redesigned Win Probability Gauge (Bright Team Arc, Muted Track)
# -----------------------------------------------------------------------------
def create_win_gauge(
    winner_name: str,
    win_probability: float,
    team1_name: str,
    team2_name: str,
    t1_prob: float,
    t2_prob: float,
) -> go.Figure:
    """Create a high-contrast circular gauge with bright franchise arc on muted track."""
    team_meta = get_team_info(winner_name)
    win_color = team_meta["primary"]
    # If franchise color is very dark (like GT #1B2133), use vibrant secondary for arc contrast
    if winner_name == "Gujarat Titans":
        arc_color = "#38BDF8"  # High-contrast cyan accent for visibility
    else:
        arc_color = win_color

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=win_probability,
            number={
                "suffix": "%",
                "font": {
                    "size": 44,
                    "color": "#F3F4F6",
                    "family": "Inter, -apple-system, sans-serif",
                },
            },
            title={
                "text": f"<span style='font-size:11px;color:#9CA3AF;letter-spacing:0.06em;text-transform:uppercase;font-family:Inter;'>{team_meta['short']} Win Probability</span>",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "rgba(255, 255, 255, 0.2)",
                    "tickfont": {"color": "#9CA3AF", "size": 11, "family": "Inter"},
                    "tickvals": [0, 25, 50, 75, 100],
                },
                "bar": {
                    "color": arc_color,
                    "thickness": 0.32,
                },
                "bgcolor": "#161F33",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.08)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(255, 255, 255, 0.02)"},
                    {"range": [50, 100], "color": "rgba(255, 255, 255, 0.05)"},
                ],
                "threshold": {
                    "line": {"color": "#3B82F6", "width": 3},
                    "thickness": 0.65,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=40, b=20),
        height=210,
        font=dict(color="#F3F4F6", family="Inter, sans-serif"),
    )
    return fig


# -----------------------------------------------------------------------------
# 2. Tactical Model Analysis & New Confidence Thresholds
# -----------------------------------------------------------------------------
def generate_ai_summary(res: Dict) -> Tuple[str, str, List[str], str]:
    """Generate bulleted domain interpretation with verified confidence thresholds.
    
    Thresholds:
      - Low Confidence: < 55%
      - Moderate Advantage: 55% – 65%
      - High Confidence: > 65%
    
    Returns:
        (confidence_label, badge_style, bullet_points_list, disclaimer_text)
    """
    winner = res["predicted_winner"]
    win_prob = res["win_probability"]
    team1 = res["team1"]
    team2 = res["team2"]
    venue = res["venue"].split(",")[0]
    feat = res["features"]

    # Strict confidence thresholds requested by user
    if win_prob < 55.0:
        conf_label = "Low Confidence"
        badge_style = "background: rgba(156, 163, 175, 0.15); color: #9CA3AF; border: 1px solid rgba(156, 163, 175, 0.3);"
    elif win_prob <= 65.0:
        conf_label = "Moderate Advantage"
        badge_style = "background: rgba(59, 130, 246, 0.15); color: #3B82F6; border: 1px solid rgba(59, 130, 246, 0.35);"
    else:
        conf_label = "High Confidence"
        badge_style = "background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.35);"

    is_winner_t1 = (winner == team1)
    winner_short = get_team_info(winner)["short"]
    rival_name = team2 if is_winner_t1 else team1
    rival_short = get_team_info(rival_name)["short"]

    winner_form = feat["team1_form_5"] if is_winner_t1 else feat["team2_form_5"]
    rival_form = feat["team2_form_5"] if is_winner_t1 else feat["team1_form_5"]
    winner_vwr = feat["team1_venue_win_rate"] if is_winner_t1 else feat["team2_venue_win_rate"]
    rival_vwr = feat["team2_venue_win_rate"] if is_winner_t1 else feat["team1_venue_win_rate"]
    winner_is_home = feat["team1_is_home"] if is_winner_t1 else feat["team2_is_home"]

    bullets: List[str] = []

    # Point 1: Ground Mastery & Conditions
    if winner_vwr > rival_vwr + 0.04:
        bullets.append(
            f"<strong>Ground Familiarity:</strong> {winner_short} holds a superior venue win rate at <strong>{venue}</strong> ({winner_vwr*100:.1f}% vs. {rival_short}'s {rival_vwr*100:.1f}%)."
        )
    elif winner_is_home:
        bullets.append(
            f"<strong>Home Advantage:</strong> Fixture played at {winner_short}'s home turf at <strong>{venue}</strong> ({winner_vwr*100:.1f}% historical ground success)."
        )
    else:
        bullets.append(
            f"<strong>Ground Records:</strong> {winner_short} maintains a <strong>{winner_vwr*100:.1f}%</strong> competitive win rate across previous appearances at {venue}."
        )

    # Point 2: Recent Form Momentum
    if winner_form > rival_form:
        bullets.append(
            f"<strong>Form Momentum:</strong> {winner_short} leads recent 5-game trajectory with <strong>{winner_form*100:.0f}%</strong> wins compared to {rival_short}'s <strong>{rival_form*100:.0f}%</strong>."
        )
    elif winner_form >= 0.6:
        bullets.append(
            f"<strong>Positive Trajectory:</strong> {winner_short} enters fixture in solid rhythm, claiming <strong>{winner_form*100:.0f}%</strong> victories across their last 5 fixtures."
        )
    else:
        bullets.append(
            f"<strong>Recent Form:</strong> {winner_short} stands at <strong>{winner_form*100:.0f}%</strong> in the last 5 matches with calibrated rolling resilience."
        )

    # Point 3: Head-to-Head Encounters
    h2h_matches = feat["h2h_matches"]
    t1_h2h = feat["team1_h2h_win_rate"]
    winner_h2h = t1_h2h if is_winner_t1 else (1.0 - t1_h2h)
    if h2h_matches >= 3 and winner_h2h >= 0.50:
        bullets.append(
            f"<strong>Rivalry Record:</strong> {winner_short} holds the historical advantage in this matchup (<strong>{winner_h2h*100:.1f}%</strong> across {h2h_matches} fixtures)."
        )
    elif h2h_matches > 0:
        bullets.append(
            f"<strong>Head-to-Head:</strong> Contested {h2h_matches} prior games with a balanced historical win rate of <strong>{winner_h2h*100:.1f}%</strong>."
        )
    else:
        bullets.append(
            f"<strong>Career Calibration:</strong> Ensemble draws on franchise baseline rates ({winner_short} overall: <strong>{feat['team1_overall_win_rate' if is_winner_t1 else 'team2_overall_win_rate']*100:.1f}%</strong>)."
        )

    # Point 4: Toss Impact
    if res["toss_winner"] == winner:
        bullets.append(
            f"<strong>Toss Advantage:</strong> {winner_short} won the toss and chose to <strong>{res['toss_decision'].upper()}</strong>, securing early match control."
        )
    else:
        toss_winner_short = get_team_info(res["toss_winner"])["short"]
        bullets.append(
            f"<strong>Counter-Toss Factor:</strong> Despite {toss_winner_short} winning the toss, model weights favor {winner_short}'s overall tactical match-up."
        )

    disclaimer = "Probabilities are statistical model estimates, not guarantees."

    return conf_label, badge_style, bullets[:4], disclaimer


# -----------------------------------------------------------------------------
# 3. Plotly Visualizations (Dark Theme, WCAG AA Compliant)
# -----------------------------------------------------------------------------
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
            y=["H2H Wins"],
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
            y=["H2H Wins"],
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
        height=80,
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
                textfont=dict(color="#F3F4F6", size=11, family="Inter"),
            ),
            go.Bar(
                name=t2_meta["short"],
                x=categories,
                y=[t2_form_5 * 100, t2_form_10 * 100],
                marker=dict(color=t2_meta["primary"], line=dict(color="rgba(255,255,255,0.2)", width=1)),
                text=[f"{t2_form_5*100:.0f}%", f"{t2_form_10*100:.0f}%"],
                textposition="outside",
                textfont=dict(color="#F3F4F6", size=11, family="Inter"),
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
            font=dict(color="#F3F4F6", size=11),
        ),
        yaxis=dict(
            range=[0, 115],
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#9CA3AF"),
            ticksuffix="%",
        ),
        xaxis=dict(
            tickfont=dict(color="#F3F4F6", size=12),
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
                textfont=dict(color="#F3F4F6", size=11, family="Inter"),
            ),
            go.Bar(
                name=t2_meta["short"],
                x=categories,
                y=[t2_venue_wr * 100, t2_overall_wr * 100],
                marker=dict(color=t2_meta["primary"], line=dict(color="rgba(255,255,255,0.2)", width=1)),
                text=[f"{t2_venue_wr*100:.1f}%", f"{t2_overall_wr*100:.1f}%"],
                textposition="outside",
                textfont=dict(color="#F3F4F6", size=11, family="Inter"),
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
            font=dict(color="#F3F4F6", size=11),
        ),
        yaxis=dict(
            range=[0, 115],
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#9CA3AF"),
            ticksuffix="%",
        ),
        xaxis=dict(
            tickfont=dict(color="#F3F4F6", size=12),
            showgrid=False,
        ),
    )
    return fig


def create_feature_importance_chart() -> go.Figure:
    """Create a Plotly horizontal bar chart showing top Gini feature importances."""
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

    fig = px.bar(
        df_feat,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=10, b=10),
        height=340,
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", ticksuffix="%"),
        yaxis=dict(tickfont=dict(color="#F3F4F6", size=11)),
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont=dict(color="#F3F4F6"))
    return fig
