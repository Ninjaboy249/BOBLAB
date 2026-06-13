import streamlit as st
import pandas as pd
import joblib
import base64
import json
import re
import requests
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="World Cup MatchLens", page_icon="⚽", layout="wide")

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

apply_background()

st.title("⚽ World Cup MatchLens")
st.caption("An explainable AI companion for understanding international soccer matchups, built from historical results.")

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

if st.button("Explain matchup", type="primary", use_container_width=True):
    if team_a == team_b:
        st.error("Please pick two different teams.")
    else:
        row, a, b = build_match_row(team_a, team_b, neutral, major)
        proba = model.predict_proba(row)[0]
        p_a, p_draw, p_b = float(proba[0]), float(proba[1]), float(proba[2])
        comparisons = explain_probability_gap(row, a, b)

        st.subheader("🧠 AI Match Analyst")
        analyst_report = generate_ai_analyst_report(team_a, team_b, [p_a, p_draw, p_b], a, b, comparisons, neutral, major)
        st.markdown(analyst_report)

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

        st.subheader("Team stats used")
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
