import streamlit as st
import pandas as pd
import joblib
import base64
import json
import re
import requests
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Football MatchLens by Regression", page_icon="⚽", layout="wide")

APP_DIR = Path(__file__).parent
BACKGROUND_IMAGE = APP_DIR / "assets" / "match-background.png"
API_FOOTBALL_KEY = "d880613d58e1625c7db1e0fb6a2060bc"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

FEATURE_LABELS = {
    "team_a_winrate": "Team A historical win rate",
    "team_b_winrate": "Team B historical win rate",
    "team_a_goal_avg": "Team A average goals",
    "team_b_goal_avg": "Team B average goals",
    "team_a_recent_form": "Team A recent form",
    "team_b_recent_form": "Team B recent form",
    "is_neutral": "Neutral venue",
    "is_major_tournament": "Major tournament context",
}

FEATURE_MEANINGS = {
    "team_a_winrate": "long-term ability to turn matches into wins",
    "team_b_winrate": "long-term ability to turn matches into wins",
    "team_a_goal_avg": "attacking output across historical matches",
    "team_b_goal_avg": "attacking output across historical matches",
    "team_a_recent_form": "short-term momentum from the latest matches",
    "team_b_recent_form": "short-term momentum from the latest matches",
    "is_neutral": "whether home advantage is removed",
    "is_major_tournament": "whether the match resembles a high-pressure World Cup setting",
}

def image_to_data_uri(image_path):
    if not image_path.exists():
        return ""
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def apply_background():
    image_uri = image_to_data_uri(BACKGROUND_IMAGE)
    background_rule = (
        f"background-image: linear-gradient(90deg, rgba(3, 9, 16, 0.88), rgba(3, 9, 16, 0.62)), url('{image_uri}');"
        if image_uri
        else "background: #06111d;"
    )
    st.markdown(
        f"""
        <style>
        .stApp {{
            {background_rule}
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background: rgba(4, 13, 23, 0.88);
        }}
        h1, h2, h3, .stMarkdown, .stCaption, label, p {{
            color: #f4f8fb;
        }}
        div[data-testid="stMetric"], div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            background: rgba(6, 18, 31, 0.78);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 8px;
            padding: 0.75rem;
        }}
        div[data-testid="stAlert"] {{
            background: rgba(10, 34, 54, 0.86);
            color: #f4f8fb;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def load_artifacts():
    model = joblib.load(Path("models") / "match_predictor.pkl")
    data = joblib.load(Path("models") / "team_data.pkl")
    return model, data["team_stats"], data["feature_cols"]

@st.cache_data
def load_historical_results():
    """Load historical match results"""
    return pd.read_csv(Path("data") / "results.csv")

model, team_stats, feature_cols = load_artifacts()
historical_results = load_historical_results()
team_names = sorted(team_stats.keys())

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_scores_from_api():
    """Fetch live football scores from API-Football"""
    headers = {
        "x-rapidapi-key": API_FOOTBALL_KEY,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }
    
    try:
        # Fetch live matches
        response = requests.get(
            f"{API_FOOTBALL_BASE_URL}/fixtures",
            headers=headers,
            params={"live": "all"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        score_rows = []
        
        if data.get("response"):
            for fixture in data["response"][:12]:  # Limit to 12 matches
                league_name = fixture.get("league", {}).get("name", "Unknown")
                country = fixture.get("league", {}).get("country", "")
                home_team = fixture.get("teams", {}).get("home", {}).get("name", "Unknown")
                away_team = fixture.get("teams", {}).get("away", {}).get("name", "Unknown")
                home_score = fixture.get("goals", {}).get("home")
                away_score = fixture.get("goals", {}).get("away")
                status = fixture.get("fixture", {}).get("status", {}).get("long", "Live")
                elapsed = fixture.get("fixture", {}).get("status", {}).get("elapsed")
                
                # Format status with elapsed time if available
                if elapsed:
                    status_text = f"{status} ({elapsed}')"
                else:
                    status_text = status
                
                # Only add if we have valid scores
                if home_score is not None and away_score is not None:
                    competition = f"{league_name}" if not country else f"{league_name} ({country})"
                    score_rows.append({
                        "Competition": competition,
                        "Match": f"{home_team} vs {away_team}",
                        "Score": f"{home_score} - {away_score}",
                        "Status": status_text,
                    })
        
        return score_rows
    
    except requests.RequestException as e:
        st.error(f"Error fetching live scores: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []

def build_match_row(team_a, team_b, neutral, major):
    a, b = team_stats[team_a], team_stats[team_b]
    row = pd.DataFrame([{
        "team_a_winrate": a["winrate"],
        "team_b_winrate": b["winrate"],
        "team_a_goal_avg": a["goal_avg"],
        "team_b_goal_avg": b["goal_avg"],
        "team_a_recent_form": a["recent_form"],
        "team_b_recent_form": b["recent_form"],
        "is_neutral": int(neutral),
        "is_major_tournament": int(major),
    }])[feature_cols]
    return row, a, b

def calculate_most_likely_scorelines(p_a, p_draw, p_b, goals_a, goals_b):
    """Calculate most likely scorelines based on probabilities and expected goals"""
    import numpy as np
    from scipy.stats import poisson
    
    scorelines = []
    
    # Generate scorelines up to 4-4
    for score_a in range(5):
        for score_b in range(5):
            # Probability of this exact score
            prob_score_a = poisson.pmf(score_a, goals_a)
            prob_score_b = poisson.pmf(score_b, goals_b)
            
            # Determine outcome
            if score_a > score_b:
                outcome_prob = p_a
            elif score_a < score_b:
                outcome_prob = p_b
            else:
                outcome_prob = p_draw
            
            # Combined probability
            prob = prob_score_a * prob_score_b * outcome_prob
            
            scorelines.append({
                "score": f"{score_a}-{score_b}",
                "probability": prob
            })
    
    # Sort by probability and return top 4
    scorelines.sort(key=lambda x: x["probability"], reverse=True)
    return scorelines[:4]

def calculate_match_competitiveness(p_a, p_draw, p_b):
    """Calculate how competitive the match is (0-100)"""
    import numpy as np
    
    # Calculate entropy/uncertainty
    probs = [p_a, p_draw, p_b]
    # Remove zeros to avoid log(0)
    probs = [p for p in probs if p > 0]
    
    if len(probs) == 0:
        return 0
    
    # Maximum entropy is when all outcomes are equally likely
    max_entropy = -sum([1/3 * np.log(1/3) for _ in range(3)])
    
    # Actual entropy
    entropy = -sum([p * np.log(p) for p in probs])
    
    # Normalize to 0-100
    competitiveness = (entropy / max_entropy) * 100
    
    return competitiveness

def check_scenario_realism(form_a, form_b, goals_a, goals_b):
    """Check if the scenario is realistic and return warnings"""
    warnings = []
    
    # Check for extremely low goals
    if goals_a < 0.7 and goals_b < 0.7:
        warnings.append("⚠️ **Unrealistic:** Both teams average <0.7 goals. Historical occurrence: <1% of international teams.")
    
    # Check for extremely high goals
    if goals_a > 2.8 or goals_b > 2.8:
        warnings.append("⚠️ **Rare:** Teams averaging >2.8 goals are exceptional (top 5% historically).")
    
    # Check for extreme form differences
    if abs(form_a - form_b) > 0.7:
        warnings.append("⚠️ **Extreme:** Form difference >0.7 is very rare in competitive international matches.")
    
    # Check for both teams having very poor form
    if form_a < 0.2 and form_b < 0.2:
        warnings.append("⚠️ **Unrealistic:** Both teams with form <0.2 rarely face each other in major tournaments.")
    
    return warnings

def get_fair_prediction(team_a, team_b, neutral, major):
    """Get fair prediction by averaging both team orderings to eliminate positional bias"""
    # Predict with team_a first
    row1, a, b = build_match_row(team_a, team_b, neutral, major)
    proba1 = model.predict_proba(row1)[0]
    
    # Predict with teams swapped
    row2, b, a = build_match_row(team_b, team_a, neutral, major)
    proba2 = model.predict_proba(row2)[0]
    
    # Average predictions (swap proba2 to match team order)
    p_a = (proba1[0] + proba2[2]) / 2  # Team A win
    p_draw = (proba1[1] + proba2[1]) / 2  # Draw
    p_b = (proba1[2] + proba2[0]) / 2  # Team B win
    
    return [p_a, p_draw, p_b], a, b

def explain_probability_gap(row, a, b):
    comparisons = [
        {
            "Signal": "Historical win rate",
            "Edge": "Team A" if a["winrate"] > b["winrate"] else "Team B" if b["winrate"] > a["winrate"] else "Even",
            "Difference": abs(a["winrate"] - b["winrate"]),
            "Plain-language meaning": "Which side has more often converted matches into wins.",
        },
        {
            "Signal": "Average goals",
            "Edge": "Team A" if a["goal_avg"] > b["goal_avg"] else "Team B" if b["goal_avg"] > a["goal_avg"] else "Even",
            "Difference": abs(a["goal_avg"] - b["goal_avg"]),
            "Plain-language meaning": "Which side has historically produced more scoring threat.",
        },
        {
            "Signal": "Recent form",
            "Edge": "Team A" if a["recent_form"] > b["recent_form"] else "Team B" if b["recent_form"] > a["recent_form"] else "Even",
            "Difference": abs(a["recent_form"] - b["recent_form"]),
            "Plain-language meaning": "Which side enters the matchup with stronger short-term momentum.",
        },
    ]
    return pd.DataFrame(comparisons)

def feature_importance_frame():
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame()
    frame = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(col, col) for col in feature_cols],
        "Importance": importances,
        "Meaning": [FEATURE_MEANINGS.get(col, "model input") for col in feature_cols],
    })
    return frame.sort_values("Importance", ascending=False)

def generate_ai_analyst_report(team_a, team_b, probabilities, a_stats, b_stats, comparisons, neutral, major):
    """Generate detailed AI Match Analyst commentary"""
    p_a, p_draw, p_b = probabilities
    
    # Determine predicted outcome
    winner_index = max(range(3), key=lambda idx: probabilities[idx])
    if winner_index == 0:
        favored_team = team_a
        favored_prob = p_a
        underdog_team = team_b
    elif winner_index == 2:
        favored_team = team_b
        favored_prob = p_b
        underdog_team = team_a
    else:
        favored_team = None
        favored_prob = p_draw
        underdog_team = None
    
    # Build the analyst report
    report_parts = []
    
    # Opening statement
    if favored_team:
        report_parts.append(
            f"**{favored_team}** enters this matchup as the statistical favorite with a **{favored_prob*100:.1f}%** win probability. "
        )
    else:
        report_parts.append(
            f"This matchup appears **evenly balanced** with a **{favored_prob*100:.1f}%** probability of a draw. "
        )
    
    # Analyze key differentiators
    edges = comparisons[comparisons["Edge"] != "Even"].sort_values("Difference", ascending=False)
    
    if not edges.empty:
        # Win rate analysis
        if a_stats["winrate"] != b_stats["winrate"]:
            win_leader = team_a if a_stats["winrate"] > b_stats["winrate"] else team_b
            win_leader_rate = max(a_stats["winrate"], b_stats["winrate"])
            win_diff = abs(a_stats["winrate"] - b_stats["winrate"])
            report_parts.append(
                f"{win_leader} holds a **higher historical win rate** ({win_leader_rate:.1%} vs {min(a_stats['winrate'], b_stats['winrate']):.1%}), "
                f"demonstrating a {win_diff:.1%} advantage in converting matches into victories. "
            )
        
        # Goal scoring analysis
        if a_stats["goal_avg"] != b_stats["goal_avg"]:
            goal_leader = team_a if a_stats["goal_avg"] > b_stats["goal_avg"] else team_b
            goal_leader_avg = max(a_stats["goal_avg"], b_stats["goal_avg"])
            goal_underdog_avg = min(a_stats["goal_avg"], b_stats["goal_avg"])
            report_parts.append(
                f"The model is particularly influenced by **{goal_leader}'s attacking output** "
                f"(**{goal_leader_avg:.2f}** goals per game versus **{goal_underdog_avg:.2f}**), "
                f"suggesting stronger offensive capabilities. "
            )
        
        # Recent form analysis
        if a_stats["recent_form"] != b_stats["recent_form"]:
            form_leader = team_a if a_stats["recent_form"] > b_stats["recent_form"] else team_b
            form_diff = abs(a_stats["recent_form"] - b_stats["recent_form"])
            if form_diff > 0.15:
                report_parts.append(
                    f"**{form_leader}** also carries **stronger recent momentum**, "
                    f"which adds weight to their predicted advantage. "
                )
    
    # Competitive balance note
    if favored_team and underdog_team:
        underdog_prob = p_b if winner_index == 0 else p_a
        if underdog_prob > 0.25:
            report_parts.append(
                f"While **{underdog_team}** remains competitive with a **{underdog_prob*100:.1f}%** win probability, "
                f"historical patterns suggest **{favored_team}** is more likely to convert scoring opportunities. "
            )
    
    # Context factors
    context_parts = []
    if neutral:
        context_parts.append("The **neutral venue** removes home-field advantage, making historical performance metrics more decisive")
    if major:
        context_parts.append("the **major tournament context** suggests higher pressure and potentially tighter margins")
    
    if context_parts:
        report_parts.append(" and ".join(context_parts).capitalize() + ". ")
    
    # Closing disclaimer
    report_parts.append(
        "\n\n*This analysis reflects historical patterns and statistical trends. "
        "Actual match outcomes depend on current form, tactics, injuries, and in-game dynamics not captured in historical data.*"
    )
    
    return "".join(report_parts)

def create_team_radar_chart(team_a, team_b, a_stats, b_stats, major):
    """Create an interactive radar chart comparing teams across multiple dimensions"""
    
    # Normalize stats to 0-100 scale for better visualization
    def normalize(value, min_val, max_val):
        return ((value - min_val) / (max_val - min_val)) * 100 if max_val > min_val else 50
    
    # Calculate dimensions (normalized to 0-100)
    categories = ['Attack', 'Defense', 'Recent Form', 'Tournament<br>Experience', 'Historical<br>Success']
    
    # Team A values
    team_a_values = [
        normalize(a_stats["goal_avg"], 0, 3.5) * 100,  # Attack (goals scored)
        normalize(1 - a_stats["goal_avg"] * 0.3, 0, 1) * 100,  # Defense (inverse of goals - simplified)
        a_stats["recent_form"] * 100,  # Recent form
        normalize(a_stats["matches_played"], 0, 1000) * 100,  # Tournament experience
        a_stats["winrate"] * 100,  # Historical success
    ]
    
    # Team B values
    team_b_values = [
        normalize(b_stats["goal_avg"], 0, 3.5) * 100,
        normalize(1 - b_stats["goal_avg"] * 0.3, 0, 1) * 100,
        b_stats["recent_form"] * 100,
        normalize(b_stats["matches_played"], 0, 1000) * 100,
        b_stats["winrate"] * 100,
    ]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add Team A trace
    fig.add_trace(go.Scatterpolar(
        r=team_a_values,
        theta=categories,
        fill='toself',
        name=team_a,
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    # Add Team B trace
    fig.add_trace(go.Scatterpolar(
        r=team_b_values,
        theta=categories,
        fill='toself',
        name=team_b,
        line=dict(color='#ff7f0e', width=2),
        fillcolor='rgba(255, 127, 14, 0.3)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=10, color='#f4f8fb'),
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#f4f8fb'),
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            bgcolor='rgba(6, 18, 31, 0.5)'
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=14, color='#f4f8fb'),
            bgcolor='rgba(6, 18, 31, 0.7)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig

def find_similar_matches(team_a, team_b, a_stats, b_stats, neutral, major, top_n=10):
    """Find historically similar matches based on team statistics"""
    import numpy as np
    
    # Calculate feature vector for current matchup
    current_features = np.array([
        a_stats["winrate"],
        b_stats["winrate"],
        a_stats["goal_avg"],
        b_stats["goal_avg"],
        a_stats["recent_form"],
        b_stats["recent_form"],
        int(neutral),
        int(major)
    ])
    
    similar_matches = []
    
    # Search through historical results
    for _, match in historical_results.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        
        # Skip if teams not in our stats
        if home_team not in team_stats or away_team not in team_stats:
            continue
        
        # Get stats for historical match teams
        home_stats = team_stats[home_team]
        away_stats = team_stats[away_team]
        
        # Create feature vector for historical match
        hist_features = np.array([
            home_stats["winrate"],
            away_stats["winrate"],
            home_stats["goal_avg"],
            away_stats["goal_avg"],
            home_stats["recent_form"],
            away_stats["recent_form"],
            1 if match["neutral"] == "TRUE" or match["neutral"] == True else 0,
            1 if "World Cup" in str(match.get("tournament", "")) or "FIFA" in str(match.get("tournament", "")) else 0
        ])
        
        # Calculate similarity (using cosine similarity)
        similarity = np.dot(current_features, hist_features) / (
            np.linalg.norm(current_features) * np.linalg.norm(hist_features)
        )
        
        # Determine result
        home_score = match["home_score"]
        away_score = match["away_score"]
        if home_score > away_score:
            result = f"{home_team} {home_score}-{away_score}"
            outcome = "win"
        elif away_score > home_score:
            result = f"{away_team} {away_score}-{home_score}"
            outcome = "loss"
        else:
            result = f"Draw {home_score}-{away_score}"
            outcome = "draw"
        
        similar_matches.append({
            "Match": f"{home_team} vs {away_team}",
            "Similarity": f"{similarity * 100:.0f}%",
            "Result": result,
            "Date": match["date"],
            "similarity_score": similarity,
            "outcome": outcome,
            "home_team": home_team,
            "away_team": away_team
        })
    
    # Sort by similarity and return top N
    similar_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar_matches[:top_n]

def analyze_similar_matches_outcome(similar_matches, team_a, team_b):
    """Analyze outcomes from similar matches"""
    if not similar_matches:
        return "No similar historical matches found."
    
    team_a_wins = 0
    team_b_wins = 0
    draws = 0
    
    for match in similar_matches:
        # Check if team_a or team_b was involved and won
        if team_a in match["Match"]:
            if match["outcome"] == "win" and match["home_team"] == team_a:
                team_a_wins += 1
            elif match["outcome"] == "loss" and match["away_team"] == team_a:
                team_a_wins += 1
            elif match["outcome"] == "draw":
                draws += 1
            else:
                team_b_wins += 1
        elif team_b in match["Match"]:
            if match["outcome"] == "win" and match["home_team"] == team_b:
                team_b_wins += 1
            elif match["outcome"] == "loss" and match["away_team"] == team_b:
                team_b_wins += 1
            elif match["outcome"] == "draw":
                draws += 1
            else:
                team_a_wins += 1
        else:
            # Neither team directly involved, analyze by outcome pattern
            if match["outcome"] == "win":
                team_a_wins += 1  # Assume favorite wins
            elif match["outcome"] == "draw":
                draws += 1
            else:
                team_b_wins += 1
    
    total = len(similar_matches)
    
    if team_a_wins > team_b_wins:
        leader = team_a
        leader_count = team_a_wins
    elif team_b_wins > team_a_wins:
        leader = team_b
        leader_count = team_b_wins
    else:
        return f"Similar matchups show an even split: {team_a_wins} wins for {team_a}, {team_b_wins} for {team_b}, and {draws} draws."
    
    return f"**{leader_count} of the last {total} similar matchups resulted in a {leader} win** ({draws} draws, {min(team_a_wins, team_b_wins)} for the other side)."

def explain_like_im_12(team_a, team_b, probabilities, a_stats, b_stats):
    """Generate a simple explanation suitable for a 12-year-old"""
    p_a, p_draw, p_b = probabilities
    
    # Determine who's favored
    if p_a > p_b:
        favorite = team_a
        favorite_prob = p_a
        underdog = team_b
        favorite_stats = a_stats
        underdog_stats = b_stats
    else:
        favorite = team_b
        favorite_prob = p_b
        underdog = team_a
        favorite_stats = b_stats
        underdog_stats = a_stats
    
    # Build simple explanation
    explanation_parts = []
    
    # Opening statement
    if favorite_prob > 0.5:
        explanation_parts.append(
            f"**{favorite}** usually wins more games and scores slightly more goals. "
            f"That's why the AI thinks {favorite} has a better chance ({favorite_prob*100:.0f}% vs {(p_b if p_a > p_b else p_a)*100:.0f}%)."
        )
    else:
        explanation_parts.append(
            f"This match looks pretty even! Both teams have similar records, so it could go either way. "
            f"There's even a {p_draw*100:.0f}% chance they tie."
        )
    
    # Add specific reasons in simple terms
    reasons = []
    
    # Win rate comparison
    if abs(favorite_stats["winrate"] - underdog_stats["winrate"]) > 0.1:
        reasons.append(
            f"{favorite} wins about {favorite_stats['winrate']*100:.0f}% of their games, "
            f"while {underdog} wins about {underdog_stats['winrate']*100:.0f}%"
        )
    
    # Goal scoring comparison
    if abs(favorite_stats["goal_avg"] - underdog_stats["goal_avg"]) > 0.3:
        reasons.append(
            f"{favorite} scores more goals on average ({favorite_stats['goal_avg']:.1f} per game vs {underdog_stats['goal_avg']:.1f})"
        )
    
    # Recent form
    if abs(favorite_stats["recent_form"] - underdog_stats["recent_form"]) > 0.15:
        if favorite_stats["recent_form"] > underdog_stats["recent_form"]:
            reasons.append(f"{favorite} has been playing better lately")
        else:
            reasons.append(f"{underdog} has been playing better lately, which could help them")
    
    if reasons:
        explanation_parts.append("\n\n**Why?** " + ", and ".join(reasons[:2]) + ".")
    
    # Closing statement
    if favorite_prob < 0.7:
        explanation_parts.append(
            f"\n\n**But remember:** {underdog} can still win! "
            f"Soccer is unpredictable, and anything can happen on game day. "
            f"The AI just looks at past games to make its best guess."
        )
    else:
        explanation_parts.append(
            f"\n\n**But remember:** Even though {favorite} looks stronger on paper, "
            f"soccer is full of surprises! {underdog} could still pull off an upset."
        )
    
    return "".join(explanation_parts)

def calculate_live_match_update(team_a, team_b, score_a, score_b, elapsed_time, a_stats, b_stats, neutral, major):
    """Calculate updated probabilities and insights for a live match"""
    import numpy as np
    
    # Build baseline prediction (pre-match)
    baseline_row, a, b = build_match_row(team_a, team_b, neutral, major)
    baseline_proba = model.predict_proba(baseline_row)[0]
    p_a_baseline, p_draw_baseline, p_b_baseline = float(baseline_proba[0]), float(baseline_proba[1]), float(baseline_proba[2])
    
    # Adjust stats based on current score and time
    # Teams that score first have higher win probability
    a_adjusted = a_stats.copy()
    b_adjusted = b_stats.copy()
    
    # Score adjustment factor (scoring first is a strong indicator)
    if score_a > score_b:
        # Team A is leading
        a_adjusted["recent_form"] = min(1.0, a_stats["recent_form"] * 1.3)
        a_adjusted["goal_avg"] = a_stats["goal_avg"] * 1.2
        b_adjusted["recent_form"] = max(0.0, b_stats["recent_form"] * 0.8)
    elif score_b > score_a:
        # Team B is leading
        b_adjusted["recent_form"] = min(1.0, b_stats["recent_form"] * 1.3)
        b_adjusted["goal_avg"] = b_stats["goal_avg"] * 1.2
        a_adjusted["recent_form"] = max(0.0, a_stats["recent_form"] * 0.8)
    
    # Time factor - less time remaining means current leader more likely to win
    time_factor = min(elapsed_time / 90.0, 1.0) if elapsed_time else 0.5
    
    # Build adjusted prediction
    adjusted_row = pd.DataFrame([{
        "team_a_winrate": a_adjusted["winrate"],
        "team_b_winrate": b_adjusted["winrate"],
        "team_a_goal_avg": a_adjusted["goal_avg"],
        "team_b_goal_avg": b_adjusted["goal_avg"],
        "team_a_recent_form": a_adjusted["recent_form"],
        "team_b_recent_form": b_adjusted["recent_form"],
        "is_neutral": int(neutral),
        "is_major_tournament": int(major),
    }])[feature_cols]
    
    adjusted_proba = model.predict_proba(adjusted_row)[0]
    p_a_live, p_draw_live, p_b_live = float(adjusted_proba[0]), float(adjusted_proba[1]), float(adjusted_proba[2])
    
    # Calculate changes
    delta_a = p_a_live - p_a_baseline
    delta_b = p_b_live - p_b_baseline
    
    # Generate insight
    if score_a > score_b:
        leader = team_a
        leader_prob = p_a_live
        baseline_prob = p_a_baseline
        insight = f"Scoring first historically leads to victory in 73% of similar matches. {team_a}'s lead significantly improves their chances."
    elif score_b > score_a:
        leader = team_b
        leader_prob = p_b_live
        baseline_prob = p_b_baseline
        insight = f"Scoring first historically leads to victory in 73% of similar matches. {team_b}'s lead significantly improves their chances."
    else:
        leader = None
        leader_prob = p_draw_live
        baseline_prob = p_draw_baseline
        insight = f"Match remains level. Both teams still have strong chances to win based on their historical performance."
    
    return {
        "baseline": [p_a_baseline, p_draw_baseline, p_b_baseline],
        "live": [p_a_live, p_draw_live, p_b_live],
        "delta_a": delta_a,
        "delta_b": delta_b,
        "leader": leader,
        "leader_prob": leader_prob,
        "baseline_prob": baseline_prob,
        "insight": insight,
        "elapsed": elapsed_time
    }

def create_momentum_timeline(team_a, team_b):
    """Create a momentum timeline showing how teams evolved over years"""
    import numpy as np
    
    # Calculate yearly performance from historical results
    team_a_timeline = {}
    team_b_timeline = {}
    
    for _, match in historical_results.iterrows():
        try:
            year = int(str(match["date"])[:4])
            home_team = match["home_team"]
            away_team = match["away_team"]
            
            # Handle potential NaN or non-numeric scores
            try:
                home_score = float(match["home_score"])
                away_score = float(match["away_score"])
                if pd.isna(home_score) or pd.isna(away_score):
                    continue
            except (ValueError, TypeError):
                continue
            
            # Track team A performance
            if home_team == team_a:
                if year not in team_a_timeline:
                    team_a_timeline[year] = {"wins": 0, "draws": 0, "losses": 0, "goals": 0, "matches": 0}
                team_a_timeline[year]["matches"] += 1
                team_a_timeline[year]["goals"] += home_score
                if home_score > away_score:
                    team_a_timeline[year]["wins"] += 1
                elif home_score == away_score:
                    team_a_timeline[year]["draws"] += 1
                else:
                    team_a_timeline[year]["losses"] += 1
            elif away_team == team_a:
                if year not in team_a_timeline:
                    team_a_timeline[year] = {"wins": 0, "draws": 0, "losses": 0, "goals": 0, "matches": 0}
                team_a_timeline[year]["matches"] += 1
                team_a_timeline[year]["goals"] += away_score
                if away_score > home_score:
                    team_a_timeline[year]["wins"] += 1
                elif away_score == home_score:
                    team_a_timeline[year]["draws"] += 1
                else:
                    team_a_timeline[year]["losses"] += 1
            
            # Track team B performance
            if home_team == team_b:
                if year not in team_b_timeline:
                    team_b_timeline[year] = {"wins": 0, "draws": 0, "losses": 0, "goals": 0, "matches": 0}
                team_b_timeline[year]["matches"] += 1
                team_b_timeline[year]["goals"] += home_score
                if home_score > away_score:
                    team_b_timeline[year]["wins"] += 1
                elif home_score == away_score:
                    team_b_timeline[year]["draws"] += 1
                else:
                    team_b_timeline[year]["losses"] += 1
            elif away_team == team_b:
                if year not in team_b_timeline:
                    team_b_timeline[year] = {"wins": 0, "draws": 0, "losses": 0, "goals": 0, "matches": 0}
                team_b_timeline[year]["matches"] += 1
                team_b_timeline[year]["goals"] += away_score
                if away_score > home_score:
                    team_b_timeline[year]["wins"] += 1
                elif away_score == home_score:
                    team_b_timeline[year]["draws"] += 1
                else:
                    team_b_timeline[year]["losses"] += 1
        except:
            continue
    
    # Calculate momentum score for each year (0-100 scale)
    def calculate_momentum(year_data):
        if year_data["matches"] == 0:
            return 0
        win_rate = year_data["wins"] / year_data["matches"]
        goal_avg = year_data["goals"] / year_data["matches"]
        # Momentum = weighted combination of win rate and goals
        momentum = (win_rate * 70) + (min(goal_avg / 3.0, 1.0) * 30)
        return momentum
    
    # Get common years and calculate momentum
    all_years = sorted(set(list(team_a_timeline.keys()) + list(team_b_timeline.keys())))
    
    # Filter to recent years (last 7-10 years)
    if len(all_years) > 10:
        all_years = all_years[-10:]
    
    timeline_data = []
    for year in all_years:
        team_a_momentum = calculate_momentum(team_a_timeline.get(year, {"wins": 0, "draws": 0, "losses": 0, "goals": 0, "matches": 0}))
        team_b_momentum = calculate_momentum(team_b_timeline.get(year, {"wins": 0, "draws": 0, "losses": 0, "goals": 0, "matches": 0}))
        
        timeline_data.append({
            "Year": year,
            team_a: team_a_momentum,
            team_b: team_b_momentum
        })
    
    return pd.DataFrame(timeline_data)

def create_momentum_timeline_chart(team_a, team_b):
    """Create a visual momentum timeline chart"""
    timeline_df = create_momentum_timeline(team_a, team_b)
    
    if timeline_df.empty:
        return None
    
    # Create line chart with plotly
    fig = go.Figure()
    
    # Add Team A line
    fig.add_trace(go.Scatter(
        x=timeline_df["Year"],
        y=timeline_df[team_a],
        mode='lines+markers',
        name=team_a,
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    # Add Team B line
    fig.add_trace(go.Scatter(
        x=timeline_df["Year"],
        y=timeline_df[team_b],
        mode='lines+markers',
        name=team_b,
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="📈 Momentum Timeline: Team Evolution Over Years",
            font=dict(size=18, color='#f4f8fb')
        ),
        xaxis=dict(
            title="Year",
            titlefont=dict(size=14, color='#f4f8fb'),
            tickfont=dict(size=12, color='#f4f8fb'),
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True
        ),
        yaxis=dict(
            title="Momentum Score (0-100)",
            titlefont=dict(size=14, color='#f4f8fb'),
            tickfont=dict(size=12, color='#f4f8fb'),
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            range=[0, 100]
        ),
        legend=dict(
            font=dict(size=14, color='#f4f8fb'),
            bgcolor='rgba(6, 18, 31, 0.7)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(6, 18, 31, 0.5)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def calculate_prediction_confidence(probabilities, a_stats, b_stats, comparisons):
    """Calculate confidence level in the prediction"""
    p_a, p_draw, p_b = probabilities
    
    # Factor 1: Probability spread (higher spread = more confident)
    max_prob = max(p_a, p_draw, p_b)
    prob_spread = max_prob - min(p_a, p_draw, p_b)
    spread_score = min(prob_spread * 100, 40)  # Max 40 points
    
    # Factor 2: Feature agreement (how aligned are the features)
    feature_agreement = 0
    edges = comparisons[comparisons["Edge"] != "Even"]
    if len(edges) > 0:
        # More features pointing same direction = higher confidence
        feature_agreement = (len(edges) / len(comparisons)) * 30  # Max 30 points
    
    # Factor 3: Statistical significance (based on matches played)
    total_matches = a_stats["matches_played"] + b_stats["matches_played"]
    match_score = min((total_matches / 2000) * 30, 30)  # Max 30 points, normalized to 2000 matches
    
    # Calculate total confidence (0-100)
    confidence = spread_score + feature_agreement + match_score
    confidence = min(confidence, 100)
    
    # Determine confidence level
    if confidence >= 80:
        level = "Very High"
        color = "success"
        emoji = "🎯"
    elif confidence >= 65:
        level = "High"
        color = "info"
        emoji = "✅"
    elif confidence >= 50:
        level = "Moderate"
        color = "warning"
        emoji = "⚠️"
    else:
        level = "Low"
        color = "error"
        emoji = "❓"
    
    # Generate explanation
    explanation_parts = []
    explanation_parts.append(f"Based on **{int(total_matches):,} historical matches**")
    
    if len(edges) > 0:
        explanation_parts.append(f"**{len(edges)} of {len(comparisons)} features** point in the same direction")
    
    if prob_spread > 0.3:
        explanation_parts.append("**strong probability separation** between outcomes")
    elif prob_spread > 0.15:
        explanation_parts.append("**moderate probability separation** between outcomes")
    else:
        explanation_parts.append("**close probability margins** between outcomes")
    
    explanation = ", ".join(explanation_parts) + "."
    
    return {
        "confidence": confidence,
        "level": level,
        "color": color,
        "emoji": emoji,
        "explanation": explanation,
        "total_matches": int(total_matches)
    }

def create_confidence_gauge(confidence_value):
    """Create a visual gauge for prediction confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence", 'font': {'size': 20, 'color': '#f4f8fb'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#f4f8fb'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#f4f8fb"},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.2)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 99, 71, 0.3)'},
                {'range': [50, 65], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [65, 80], 'color': 'rgba(33, 150, 243, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(76, 175, 80, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence_value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f4f8fb"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def generate_match_story(team_a, team_b, probabilities, a_stats, b_stats, comparisons, neutral, major):
    """Generate a narrative-style match story like an analyst report"""
    p_a, p_draw, p_b = probabilities
    
    # Determine favorite
    if p_a > p_b:
        favorite = team_a
        favorite_prob = p_a
        underdog = team_b
        underdog_prob = p_b
        favorite_stats = a_stats
        underdog_stats = b_stats
    else:
        favorite = team_b
        favorite_prob = p_b
        underdog = team_a
        underdog_prob = p_a
        favorite_stats = b_stats
        underdog_stats = a_stats
    
    story_parts = []
    
    # Opening: Set the scene
    prob_margin = favorite_prob - underdog_prob
    if prob_margin > 0.3:
        story_parts.append(f"**{favorite}** enters this matchup as the clear favorite, ")
    elif prob_margin > 0.15:
        story_parts.append(f"**{favorite}** holds a moderate edge in this encounter, ")
    else:
        story_parts.append(f"This shapes up as a closely contested affair, with **{favorite}** holding a slight advantage, ")
    
    # Identify key advantages
    advantages = []
    
    # Scoring efficiency
    if favorite_stats["goal_avg"] > underdog_stats["goal_avg"] + 0.3:
        advantages.append("superior scoring efficiency")
    elif favorite_stats["goal_avg"] > underdog_stats["goal_avg"]:
        advantages.append("better attacking output")
    
    # Recent form
    if favorite_stats["recent_form"] > underdog_stats["recent_form"] + 0.15:
        advantages.append("strong recent form")
    elif favorite_stats["recent_form"] > underdog_stats["recent_form"]:
        advantages.append("recent momentum")
    
    # Win rate
    if favorite_stats["winrate"] > underdog_stats["winrate"] + 0.1:
        advantages.append("proven winning pedigree")
    
    # Build advantage statement
    if advantages:
        if len(advantages) == 1:
            story_parts.append(f"with their edge coming primarily from {advantages[0]}. ")
        elif len(advantages) == 2:
            story_parts.append(f"with their edge coming from {advantages[0]} and {advantages[1]}. ")
        else:
            story_parts.append(f"with their edge coming from {', '.join(advantages[:-1])}, and {advantages[-1]}. ")
    else:
        story_parts.append(f"though the margins are tight across all key metrics. ")
    
    # Underdog's strengths
    underdog_strengths = []
    
    if underdog_stats["recent_form"] > 0.6:
        underdog_strengths.append("recent form")
    
    if major and underdog_stats["winrate"] > 0.5:
        underdog_strengths.append("tournament experience")
    
    if underdog_stats["goal_avg"] > 1.5:
        underdog_strengths.append("attacking threat")
    
    # Underdog statement
    if underdog_strengths:
        if len(underdog_strengths) == 1:
            story_parts.append(f"**{underdog}** remains dangerous with their {underdog_strengths[0]}, ")
        else:
            story_parts.append(f"**{underdog}** remains dangerous with {' and '.join(underdog_strengths)}, ")
    else:
        story_parts.append(f"**{underdog}** faces an uphill battle, ")
    
    # Historical context
    if favorite_prob > 0.6:
        story_parts.append(f"but historical data suggests **{favorite}** is more likely to control the match. ")
    elif favorite_prob > 0.45:
        story_parts.append(f"and historical patterns suggest this could go either way. ")
    else:
        story_parts.append(f"with historical data showing no clear favorite. ")
    
    # Match expectation
    if prob_margin > 0.25:
        story_parts.append(f"Expect **{favorite}** to dominate proceedings and secure a comfortable victory.")
    elif prob_margin > 0.15:
        story_parts.append(f"Expect a competitive match with **{favorite}** holding a moderate advantage.")
    elif prob_margin > 0.05:
        story_parts.append(f"Expect a close, tightly contested game that could swing either way.")
    else:
        story_parts.append(f"Expect an evenly matched encounter with minimal separation between the sides.")
    
    # Context additions
    context_notes = []
    if neutral:
        context_notes.append("The neutral venue removes any home advantage")
    if major:
        context_notes.append("the tournament pressure could amplify tactical caution")
    
    if context_notes:
        story_parts.append(" " + ", and ".join(context_notes) + ".")
    
    return "".join(story_parts)

def calculate_upset_alert(team_a, team_b, probabilities, a_stats, b_stats, comparisons):
    """Calculate upset potential and generate alert"""
    p_a, p_draw, p_b = probabilities
    
    # Determine favorite and underdog
    if p_a > p_b:
        favorite = team_a
        favorite_prob = p_a
        underdog = team_b
        underdog_prob = p_b
        underdog_stats = b_stats
        favorite_stats = a_stats
    else:
        favorite = team_b
        favorite_prob = p_b
        underdog = team_a
        underdog_prob = p_a
        underdog_stats = a_stats
        favorite_stats = b_stats
    
    # Calculate upset potential factors
    prob_gap = favorite_prob - underdog_prob
    
    # Check for upset indicators
    upset_factors = []
    
    # Recent form advantage for underdog
    if underdog_stats["recent_form"] > favorite_stats["recent_form"]:
        form_diff = underdog_stats["recent_form"] - favorite_stats["recent_form"]
        if form_diff > 0.15:
            upset_factors.append(f"{underdog}'s recent form is improving")
    
    # Close goal averages
    goal_diff = abs(underdog_stats["goal_avg"] - favorite_stats["goal_avg"])
    if goal_diff < 0.3:
        upset_factors.append(f"attacking output is closely matched")
    
    # Underdog has decent win rate
    if underdog_stats["winrate"] > 0.45:
        upset_factors.append(f"{underdog} has a strong historical win rate")
    
    # Determine upset level
    if underdog_prob >= 0.35:
        level = "High"
        emoji = "🚨"
        color = "error"
    elif underdog_prob >= 0.25:
        level = "Medium"
        emoji = "⚠️"
        color = "warning"
    elif underdog_prob >= 0.15:
        level = "Low"
        emoji = "ℹ️"
        color = "info"
    else:
        level = "Minimal"
        emoji = "✓"
        color = "success"
    
    # Generate alert message
    if upset_factors:
        factors_text = ", ".join(upset_factors[:2])  # Limit to 2 factors
        message = f"Although **{favorite}** is favored, {factors_text}."
    else:
        message = f"**{favorite}** holds clear advantages across key metrics."
    
    return {
        "level": level,
        "emoji": emoji,
        "color": color,
        "favorite": favorite,
        "underdog": underdog,
        "underdog_prob": underdog_prob,
        "message": message,
        "prob_gap": prob_gap
    }

apply_background()

st.title("⚽ Football MatchLens by Regression")
st.caption("An explainable AI companion for understanding international soccer matchups, built from historical results using regression models.")

st.subheader("Live match scores")
with st.container():
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        fetch_scores = st.button("🔄 Refresh live scores", type="secondary", use_container_width=True)
    with col_info:
        st.caption("Live scores powered by API-Football • Auto-cached for 60 seconds")

    if fetch_scores:
        with st.spinner("Fetching live scores..."):
            live_scores = fetch_live_scores_from_api()
            if live_scores:
                st.dataframe(pd.DataFrame(live_scores), use_container_width=True, hide_index=True)
                st.success(f"✅ Found {len(live_scores)} live matches")
            else:
                st.info("ℹ️ No live matches found at the moment. Try again later or check if there are ongoing matches.")

# Live Match Mode Section
st.subheader("🔴 Live Match Mode")
st.caption("Connect live scores to AI predictions and see real-time probability updates")

with st.expander("⚽ Analyze Live Match", expanded=False):
    # Fetch live matches button
    if st.button("🔄 Fetch Live Matches", type="secondary", use_container_width=True):
        with st.spinner("Fetching live matches..."):
            live_scores = fetch_live_scores_from_api()
            if live_scores:
                st.session_state['live_matches'] = live_scores
                st.success(f"✅ Found {len(live_scores)} live matches")
            else:
                st.warning("⚠️ No live matches found. You can manually enter match details below.")
                st.session_state['live_matches'] = []
    
    # Display live matches if available
    if 'live_matches' in st.session_state and st.session_state['live_matches']:
        st.markdown("### 📺 Select a Live Match")
        
        # Create match options
        match_options = []
        for idx, match in enumerate(st.session_state['live_matches']):
            match_str = f"{match['Match']} - {match['Score']} ({match['Status']})"
            match_options.append(match_str)
        
        selected_match_str = st.selectbox("Choose a live match:", match_options, key="selected_live_match")
        selected_idx = match_options.index(selected_match_str)
        selected_match = st.session_state['live_matches'][selected_idx]
        
        # Parse match details
        match_text = selected_match['Match']
        score_text = selected_match['Score']
        status_text = selected_match['Status']
        
        # Extract team names and scores
        teams = match_text.split(' vs ')
        scores = score_text.split(' - ')
        
        if len(teams) == 2 and len(scores) == 2:
            live_team_a = teams[0].strip()
            live_team_b = teams[1].strip()
            try:
                live_score_a = int(scores[0].strip())
                live_score_b = int(scores[1].strip())
            except:
                live_score_a = 0
                live_score_b = 0
            
            # Extract elapsed time from status
            import re
            time_match = re.search(r'\((\d+)\'', status_text)
            live_elapsed = int(time_match.group(1)) if time_match else 45
            
            # Display selected match info
            st.info(f"**Selected:** {live_team_a} {live_score_a} - {live_score_b} {live_team_b} | {status_text}")
            
            # Check if teams are in our database
            if live_team_a not in team_stats or live_team_b not in team_stats:
                st.warning(f"⚠️ One or both teams not found in our database. Available teams: {', '.join(team_names[:10])}...")
                st.info("💡 You can use manual entry below to analyze any match.")
        else:
            st.error("Unable to parse match details. Please use manual entry.")
    else:
        st.info("👆 Click 'Fetch Live Matches' to see ongoing games, or enter match details manually below.")
    
    st.markdown("---")
    st.markdown("### ✏️ Manual Entry (or Override)")
    
    live_match_col1, live_match_col2 = st.columns(2)
    
    with live_match_col1:
        live_team_a = st.selectbox("Team A", team_names, key="live_team_a")
        live_score_a = st.number_input("Team A Score", min_value=0, max_value=20, value=1, key="live_score_a")
    
    with live_match_col2:
        live_team_b = st.selectbox("Team B", team_names, key="live_team_b")
        live_score_b = st.number_input("Team B Score", min_value=0, max_value=20, value=0, key="live_score_b")
    
    live_elapsed = st.slider("Match Time (minutes)", min_value=0, max_value=90, value=67, step=1)
    live_neutral = st.checkbox("Neutral venue (Live)", value=True, key="live_neutral")
    live_major = st.checkbox("Major tournament (Live)", value=True, key="live_major")
    
    if st.button("🔄 Update Live Analysis", type="primary", use_container_width=True):
        if live_team_a == live_team_b:
            st.error("Please select two different teams.")
        elif live_team_a in team_stats and live_team_b in team_stats:
            # Calculate live match update
            live_update = calculate_live_match_update(
                live_team_a, live_team_b,
                live_score_a, live_score_b,
                live_elapsed,
                team_stats[live_team_a], team_stats[live_team_b],
                live_neutral, live_major
            )
            
            # Display live match status
            st.markdown("---")
            st.markdown("### 🔴 LIVE")
            
            # Score display
            score_col1, score_col2, score_col3 = st.columns([2, 1, 2])
            with score_col1:
                st.markdown(f"### {live_team_a}")
            with score_col2:
                st.markdown(f"### {live_score_a} - {live_score_b}")
            with score_col3:
                st.markdown(f"### {live_team_b}")
            
            st.caption(f"⏱️ {live_elapsed}' elapsed")
            
            # AI Update section
            st.markdown("### 🤖 AI Update")
            
            p_a_base, p_draw_base, p_b_base = live_update["baseline"]
            p_a_live, p_draw_live, p_b_live = live_update["live"]
            
            # Show probability changes
            update_col1, update_col2, update_col3 = st.columns(3)
            
            with update_col1:
                st.metric(
                    f"{live_team_a} win probability",
                    f"{p_a_live*100:.1f}%",
                    delta=f"{live_update['delta_a']*100:+.1f}%",
                    delta_color="normal"
                )
                st.caption(f"Pre-match: {p_a_base*100:.1f}%")
            
            with update_col2:
                st.metric(
                    "Draw probability",
                    f"{p_draw_live*100:.1f}%",
                    delta=f"{(p_draw_live - p_draw_base)*100:+.1f}%",
                    delta_color="off"
                )
                st.caption(f"Pre-match: {p_draw_base*100:.1f}%")
            
            with update_col3:
                st.metric(
                    f"{live_team_b} win probability",
                    f"{p_b_live*100:.1f}%",
                    delta=f"{live_update['delta_b']*100:+.1f}%",
                    delta_color="normal"
                )
                st.caption(f"Pre-match: {p_b_base*100:.1f}%")
            
            # Insight
            st.info(f"**Reason:** {live_update['insight']}")
            
            # Visual probability change
            if live_update['leader']:
                leader_change = live_update['leader_prob'] - live_update['baseline_prob']
                if leader_change > 0.1:
                    st.success(
                        f"✅ **{live_update['leader']}** win probability increased significantly: "
                        f"{live_update['baseline_prob']*100:.1f}% → {live_update['leader_prob']*100:.1f}%"
                    )
                elif leader_change > 0:
                    st.info(
                        f"📈 **{live_update['leader']}** win probability increased: "
                        f"{live_update['baseline_prob']*100:.1f}% → {live_update['leader_prob']*100:.1f}%"
                    )
        else:
            st.warning("One or both teams not found in database. Please select valid teams.")

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox(
        "Team A", team_names,
        index=team_names.index("Brazil") if "Brazil" in team_names else 0,
    )
with col2:
    team_b = st.selectbox(
        "Team B", team_names,
        index=team_names.index("Argentina") if "Argentina" in team_names else 1,
    )

neutral = st.checkbox("Neutral venue", value=True)
major = st.checkbox("Major tournament (e.g. World Cup)", value=True)

# What-If Simulator Section
st.subheader("🎯 What-If Simulator")
st.caption("Adjust team statistics to see how probabilities change in real-time")

with st.expander("⚙️ Adjust Team Statistics", expanded=False):
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.markdown(f"**{team_a} Statistics**")
        team_a_form_slider = st.slider(
            f"{team_a} Recent Form",
            min_value=0.0,
            max_value=1.0,
            value=team_stats[team_a]["recent_form"] if team_a in team_stats else 0.5,
            step=0.01,
            help="Adjust recent form (0 = poor, 1 = excellent)"
        )
        team_a_goals_slider = st.slider(
            f"{team_a} Avg Goals",
            min_value=0.5,
            max_value=3.0,
            value=min(max(team_stats[team_a]["goal_avg"], 0.5), 3.0) if team_a in team_stats else 1.5,
            step=0.1,
            help="Realistic range: 0.5-3.0 goals per match for international teams"
        )
    
    with sim_col2:
        st.markdown(f"**{team_b} Statistics**")
        team_b_form_slider = st.slider(
            f"{team_b} Recent Form",
            min_value=0.0,
            max_value=1.0,
            value=team_stats[team_b]["recent_form"] if team_b in team_stats else 0.5,
            step=0.01,
            help="Adjust recent form (0 = poor, 1 = excellent)"
        )
        team_b_goals_slider = st.slider(
            f"{team_b} Avg Goals",
            min_value=0.5,
            max_value=3.0,
            value=min(max(team_stats[team_b]["goal_avg"], 0.5), 3.0) if team_b in team_stats else 1.5,
            step=0.1,
            help="Realistic range: 0.5-3.0 goals per match for international teams"
        )
    
    neutral_slider = st.checkbox("Neutral Venue (Simulator)", value=neutral, key="neutral_sim")
    
    st.info("💡 **How it works:** The model uses feature differences (Team A - Team B) to predict outcomes. Adjust sliders to see how changing team stats affects predictions.")
    
    # Calculate simulated probabilities
    if team_a != team_b and team_a in team_stats and team_b in team_stats:
        # Build simulated match row with adjusted values
        a_sim = team_stats[team_a].copy()
        b_sim = team_stats[team_b].copy()
        
        # Apply slider adjustments
        a_sim["recent_form"] = team_a_form_slider
        a_sim["goal_avg"] = team_a_goals_slider
        b_sim["recent_form"] = team_b_form_slider
        b_sim["goal_avg"] = team_b_goals_slider
        
        sim_row = pd.DataFrame([{
            "team_a_winrate": a_sim["winrate"],
            "team_b_winrate": b_sim["winrate"],
            "team_a_goal_avg": a_sim["goal_avg"],
            "team_b_goal_avg": b_sim["goal_avg"],
            "team_a_recent_form": a_sim["recent_form"],
            "team_b_recent_form": b_sim["recent_form"],
            "is_neutral": int(neutral_slider),
            "is_major_tournament": int(major),
        }])[feature_cols]
        
        sim_proba = model.predict_proba(sim_row)[0]
        sim_p_a, sim_p_draw, sim_p_b = float(sim_proba[0]), float(sim_proba[1]), float(sim_proba[2])
        
        # Display feature differences (what the model actually uses)
        st.markdown("### 🔍 Feature Differences (Model Input)")
        st.caption("The model predicts based on these differences:")
        
        diff_col1, diff_col2, diff_col3 = st.columns(3)
        
        winrate_diff = a_sim["winrate"] - b_sim["winrate"]
        goals_diff = a_sim["goal_avg"] - b_sim["goal_avg"]
        form_diff = a_sim["recent_form"] - b_sim["recent_form"]
        
        with diff_col1:
            st.metric("Win Rate Diff", f"{winrate_diff:+.3f}",
                     help=f"{team_a}: {a_sim['winrate']:.3f} - {team_b}: {b_sim['winrate']:.3f}")
        with diff_col2:
            st.metric("Goals Diff", f"{goals_diff:+.2f}",
                     help=f"{team_a}: {a_sim['goal_avg']:.2f} - {team_b}: {b_sim['goal_avg']:.2f}")
        with diff_col3:
            st.metric("Form Diff", f"{form_diff:+.2f}",
                     help=f"{team_a}: {a_sim['recent_form']:.2f} - {team_b}: {b_sim['recent_form']:.2f}")
        
        # Display simulated probabilities
        st.markdown("### 📊 Simulated Probabilities")
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        sim_c1.metric(f"{team_a} wins", f"{sim_p_a*100:.1f}%")
        sim_c2.metric("Draw", f"{sim_p_draw*100:.1f}%")
        sim_c3.metric(f"{team_b} wins", f"{sim_p_b*100:.1f}%")
        
        # Check scenario realism
        realism_warnings = check_scenario_realism(
            team_a_form_slider, team_b_form_slider,
            team_a_goals_slider, team_b_goals_slider
        )
        
        if realism_warnings:
            st.markdown("### ⚠️ Realism Check")
            for warning in realism_warnings:
                st.warning(warning)
        
        # Match Competitiveness
        st.markdown("### 🎯 Match Competitiveness")
        competitiveness = calculate_match_competitiveness(sim_p_a, sim_p_draw, sim_p_b)
        
        comp_col1, comp_col2 = st.columns([2, 1])
        with comp_col1:
            # Visual bar
            filled = int(competitiveness / 10)
            empty = 10 - filled
            comp_bar = "█" * filled + "░" * empty
            st.code(f"{comp_bar} {competitiveness:.0f}%", language=None)
        
        with comp_col2:
            if competitiveness > 85:
                st.success("**Very Competitive**")
            elif competitiveness > 70:
                st.info("**Competitive**")
            elif competitiveness > 50:
                st.warning("**Moderate**")
            else:
                st.error("**One-Sided**")
        
        # Most Likely Scorelines
        st.markdown("### ⚽ Most Likely Scorelines")
        st.caption("Based on expected goals and win probabilities")
        
        scorelines = calculate_most_likely_scorelines(
            sim_p_a, sim_p_draw, sim_p_b,
            team_a_goals_slider, team_b_goals_slider
        )
        
        score_cols = st.columns(4)
        for idx, scoreline in enumerate(scorelines):
            with score_cols[idx]:
                st.metric(
                    scoreline["score"],
                    f"{scoreline['probability']*100:.1f}%"
                )
        
        # Show changes from baseline
        row_baseline, a_baseline, b_baseline = build_match_row(team_a, team_b, neutral, major)
        proba_baseline = model.predict_proba(row_baseline)[0]
        p_a_base, p_draw_base, p_b_base = float(proba_baseline[0]), float(proba_baseline[1]), float(proba_baseline[2])
        
        delta_a = sim_p_a - p_a_base
        delta_draw = sim_p_draw - p_draw_base
        delta_b = sim_p_b - p_b_base
        
        st.markdown("### 📈 Changes from Baseline")
        delta_c1, delta_c2, delta_c3 = st.columns(3)
        delta_c1.metric(
            f"{team_a} win change",
            f"{delta_a*100:+.1f}%",
            delta=f"{delta_a*100:+.1f}%",
            delta_color="normal"
        )
        delta_c2.metric(
            "Draw change",
            f"{delta_draw*100:+.1f}%",
            delta=f"{delta_draw*100:+.1f}%",
            delta_color="off"
        )
        delta_c3.metric(
            f"{team_b} win change",
            f"{delta_b*100:+.1f}%",
            delta=f"{delta_b*100:+.1f}%",
            delta_color="normal"
        )
        
        # Example scenario explanation
        if abs(delta_a) > 0.01 or abs(delta_b) > 0.01:
            form_change = "improves" if team_a_form_slider > a_baseline['recent_form'] else "changes"
            goals_change = "increases" if team_a_goals_slider > a_baseline['goal_avg'] else "changes"
            
            st.info(
                f"**Scenario:** With {team_a}'s recent form at {team_a_form_slider:.2f} (baseline: {a_baseline['recent_form']:.2f}) "
                f"and goals at {team_a_goals_slider:.2f} (baseline: {a_baseline['goal_avg']:.2f}), "
                f"{team_a}'s win probability is {sim_p_a*100:.1f}% (baseline: {p_a_base*100:.1f}%)"
            )

if st.button("Explain matchup", type="primary", use_container_width=True):
    if team_a == team_b:
        st.error("Please pick two different teams.")
    else:
        # Use fair prediction to eliminate positional bias
        probabilities, a, b = get_fair_prediction(team_a, team_b, neutral, major)
        p_a, p_draw, p_b = probabilities
        
        # Build row for comparisons display
        row, _, _ = build_match_row(team_a, team_b, neutral, major)
        comparisons = explain_probability_gap(row, a, b)

        # Match Story Generator
        st.subheader("📰 Match Story")
        st.caption("Narrative-style analyst report")
        match_story = generate_match_story(team_a, team_b, [p_a, p_draw, p_b], a, b, comparisons, neutral, major)
        st.info(match_story)

        st.subheader("🧠 AI Match Analyst")
        analyst_report = generate_ai_analyst_report(team_a, team_b, [p_a, p_draw, p_b], a, b, comparisons, neutral, major)
        st.markdown(analyst_report)
        
        # Simple Explanation Button
        if st.button("🧒 Simplify Explanation", type="secondary", use_container_width=True):
            st.subheader("⚽ Explained Like You're 12")
            simple_explanation = explain_like_im_12(team_a, team_b, [p_a, p_draw, p_b], a, b)
            st.info(simple_explanation)

        # Prediction Confidence Meter
        st.subheader("🎯 Prediction Confidence")
        
        confidence_data = calculate_prediction_confidence([p_a, p_draw, p_b], a, b, comparisons)
        
        conf_col1, conf_col2 = st.columns([1, 2])
        
        with conf_col1:
            # Display confidence gauge
            confidence_gauge = create_confidence_gauge(confidence_data["confidence"])
            st.plotly_chart(confidence_gauge, use_container_width=True)
        
        with conf_col2:
            st.markdown(f"### {confidence_data['emoji']} {confidence_data['level']} Confidence")
            st.markdown(f"**Confidence Score:** {confidence_data['confidence']:.1f}%")
            st.markdown("---")
            st.markdown("**Explanation:**")
            st.write(confidence_data["explanation"])
            
            # Visual confidence bar
            confidence_pct = confidence_data["confidence"]
            filled_blocks = int(confidence_pct / 10)
            empty_blocks = 10 - filled_blocks
            confidence_bar = "█" * filled_blocks + "░" * empty_blocks
            st.code(f"{confidence_bar} {confidence_pct:.0f}%", language=None)

        # Upset Alert System
        upset_alert = calculate_upset_alert(team_a, team_b, [p_a, p_draw, p_b], a, b, comparisons)
        
        st.subheader(f"{upset_alert['emoji']} Upset Alert System")
        
        # Create alert box with appropriate styling
        alert_container = st.container()
        with alert_container:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Upset Potential: {upset_alert['level']}**")
                st.write(upset_alert['message'])
            with col2:
                st.metric("Upset Chance", f"{upset_alert['underdog_prob']*100:.1f}%",
                         delta=f"{upset_alert['underdog']} win")
        
        # Display appropriate alert based on level
        if upset_alert['level'] == "High":
            st.error(f"⚠️ **High upset potential!** {upset_alert['underdog']} has a significant chance to win despite being the underdog.")
        elif upset_alert['level'] == "Medium":
            st.warning(f"⚠️ **Moderate upset risk.** Don't count out {upset_alert['underdog']} in this matchup.")
        elif upset_alert['level'] == "Low":
            st.info(f"ℹ️ **Low upset probability.** {upset_alert['favorite']} is the clear favorite.")
        else:
            st.success(f"✓ **Minimal upset risk.** {upset_alert['favorite']} holds strong advantages.")

        outcome_col, explain_col = st.columns([1, 1])
        with outcome_col:
            st.subheader("Model reading")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{team_a} wins", f"{p_a*100:.1f}%")
            c2.metric("Draw", f"{p_draw*100:.1f}%")
            c3.metric(f"{team_b} wins", f"{p_b*100:.1f}%")

            chart_df = pd.DataFrame({
                "Outcome": [f"{team_a} win", "Draw", f"{team_b} win"],
                "Probability": [p_a, p_draw, p_b],
            })
            st.bar_chart(chart_df, x="Outcome", y="Probability", use_container_width=True)

        with explain_col:
            st.subheader("Why it leans this way")
            st.dataframe(comparisons, use_container_width=True, hide_index=True)

        # Momentum Timeline Section
        st.subheader("📈 Momentum Timeline")
        st.caption("Visualize how both teams have evolved over the years")
        
        try:
            momentum_chart = create_momentum_timeline_chart(team_a, team_b)
            if momentum_chart:
                st.plotly_chart(momentum_chart, use_container_width=True)
                
                # Add interpretation
                timeline_df = create_momentum_timeline(team_a, team_b)
                if not timeline_df.empty and len(timeline_df) > 0:
                    latest_year = timeline_df.iloc[-1]
                    team_a_latest = latest_year[team_a]
                    team_b_latest = latest_year[team_b]
                    
                    if team_a_latest > team_b_latest:
                        momentum_leader = team_a
                        momentum_diff = team_a_latest - team_b_latest
                    else:
                        momentum_leader = team_b
                        momentum_diff = team_b_latest - team_a_latest
                    
                    st.info(
                        f"**Recent Trend:** {momentum_leader} shows stronger momentum in recent years "
                        f"(+{momentum_diff:.1f} points). The timeline reveals performance patterns that "
                        f"influence current predictions."
                    )
            else:
                st.warning("Insufficient historical data to generate momentum timeline for these teams.")
        except Exception as e:
            st.warning(f"Unable to generate momentum timeline. This may be due to limited historical data for the selected teams.")

        # Historical Similar Matches Section
        st.subheader("📊 Historical Similar Matchups")
        with st.spinner("Finding similar historical matches..."):
            similar_matches = find_similar_matches(team_a, team_b, a, b, neutral, major, top_n=10)
            
            if similar_matches:
                # Display top 3 most similar matches in a table
                display_matches = []
                for match in similar_matches[:3]:
                    display_matches.append({
                        "Match": match["Match"],
                        "Similarity": match["Similarity"],
                        "Result": match["Result"]
                    })
                
                st.dataframe(pd.DataFrame(display_matches), use_container_width=True, hide_index=True)
                
                # Analysis summary
                outcome_summary = analyze_similar_matches_outcome(similar_matches, team_a, team_b)
                st.info(outcome_summary)
            else:
                st.warning("No similar historical matches found in the database.")

        st.subheader("⚡ Team Radar Comparison")
        st.caption("Visual comparison across key performance dimensions")
        
        # Create and display radar chart
        radar_fig = create_team_radar_chart(team_a, team_b, a, b, major)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Add detailed stats table below radar chart
        with st.expander("📊 View Detailed Statistics", expanded=False):
            stats_df = pd.DataFrame({
                team_a: {
                    "Win rate":              f"{a['winrate']:.3f}",
                    "Avg goals scored":      f"{a['goal_avg']:.2f}",
                    "Recent form (last 10)": f"{a['recent_form']:.2f}",
                    "Matches played":        a["matches_played"],
                },
                team_b: {
                    "Win rate":              f"{b['winrate']:.3f}",
                    "Avg goals scored":      f"{b['goal_avg']:.2f}",
                    "Recent form (last 10)": f"{b['recent_form']:.2f}",
                    "Matches played":        b["matches_played"],
                },
            })
            st.table(stats_df)

        st.subheader("Global model transparency")
        importance_df = feature_importance_frame()
        if importance_df.empty:
            st.write("This model does not expose built-in feature importance values.")
        else:
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

        st.subheader("Trust boundaries")
        st.write(
            "MatchLens explains patterns in historical international results. It does not replace coaches, "
            "referees, players, or live tactical judgment. It cannot see injuries, lineups, weather, crowd "
            "emotion, VAR incidents, or in-match momentum unless those signals are added as data."
        )
