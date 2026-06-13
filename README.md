# World Cup MatchLens

World Cup MatchLens is an explainable AI companion for international soccer matchups. Instead of only predicting a result, it helps fans understand why an AI model leans toward one team, where the matchup looks balanced, and which historical signals are driving the model's reading.

This project was built for the IBM SkillsBuild AI Builders Challenge using the June football lab assets and IBM Bob as the AI-assisted development workflow.

## Problem

The World Cup is watched by billions of people, but fans do not all bring the same language, soccer background, tactical knowledge, or trust in match interpretation. A plain probability such as "Brazil 42%" does not help a new fan understand the game, and it can make AI feel opaque.

MatchLens focuses on match understanding:

- Why does the model lean toward one side?
- Which signals are most important?
- Is the matchup historically close or clearly tilted?
- What should fans trust, and what should they treat carefully?

The goal is not to replace coaches, referees, or players. The goal is to make AI interpretation more transparent and accessible.

## Solution

MatchLens is a Streamlit prototype that lets a user select two international teams and explain the matchup through:

- optional live-score fetching from LiveScore
- outcome probabilities from a trained machine learning model
- plain-language AI explanation of the model's reasoning
- side-by-side comparison of win rate, average goals, recent form, and venue context
- global model transparency through feature importance values
- trust boundaries that explain what the model can and cannot know

## AI And Technical Approach

The app uses a `RandomForestClassifier` trained on historical international football results from 1872 to 2026. The model uses engineered features for each team:

- historical win rate
- average goals scored
- recent form
- neutral venue flag
- major tournament flag

The prototype then adds an explainability layer on top of the model output. It converts model probabilities, feature comparisons, and feature importance values into fan-facing explanations. This turns the project from a pure score predictor into an AI-powered match explainer.

IBM technologies used:

- **IBM Bob**: used as the AI coding assistant and lab workflow for building and refining the football AI prototype
- **IBM SkillsBuild football lab assets**: used as the foundation for the model, dataset flow, and hands-on learning structure

## Why It Matters

During the World Cup, people often debate tactics, momentum, team quality, pressure, and fairness. Explainable AI can make those debates more inclusive by giving fans a clearer view of the evidence behind an interpretation.

MatchLens helps:

- new fans learn what common football signals mean
- experienced fans inspect the model instead of accepting a black-box answer
- multilingual or global audiences get simpler explanations of complex match context
- challenge judges see how AI can support understanding, not just prediction

## Run The Prototype

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL Streamlit provides in your browser.

## Project Structure

- `app.py` - Streamlit explainable AI prototype
- `assets/match-background.png` - stadium background image used by the app
- `models/match_predictor.pkl` - trained RandomForest match model
- `models/team_data.pkl` - team feature data used by the app
- `data/results.csv` - historical international football results dataset
- `bob_generated_code.ipynb` - notebook generated during the IBM Bob lab workflow
- `hands-on-labs/` - IBM SkillsBuild hands-on lab materials

## Current Limitations

MatchLens explains historical statistical patterns. Live scores are fetched on demand from the configured source URL, but external pages can change their structure or block automated access. The app does not currently include live lineups, injuries, substitutions, weather, referee decisions, player tracking, tactical formations, or real-time momentum. Those would be strong next steps for turning this proof of concept into a live World Cup companion.
