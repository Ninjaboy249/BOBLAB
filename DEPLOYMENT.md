# Deployment Guide for World Cup MatchLens

## Deploy to Streamlit Cloud

### Prerequisites
- GitHub account with the repository pushed
- Streamlit Cloud account (free at https://streamlit.io/cloud)

### Step-by-Step Deployment

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Select your repository: `Ninjaboy249/BOBLAB`
   - Branch: `main`
   - Main file path: `app.py`

3. **Advanced Settings (Optional)**
   - Python version: 3.9 or higher
   - No additional secrets needed (API key is in code)

4. **Deploy**
   - Click "Deploy!"
   - Wait 2-3 minutes for deployment
   - Your app will be live at: `https://[your-app-name].streamlit.app`

### Quick Deploy Link
After signing in to Streamlit Cloud, use this direct link:
```
https://share.streamlit.io/deploy?repository=Ninjaboy249/BOBLAB&branch=main&mainModule=app.py
```

### Features Included
✅ What-If Simulator with interactive sliders
✅ Team Radar Chart (5 dimensions)
✅ Simplify Explanation (ELI12)
✅ Live Match Mode with real-time updates
✅ Momentum Timeline visualization
✅ Prediction Confidence Meter
✅ Match Story Generator

### Requirements
All dependencies are in `requirements.txt`:
- streamlit
- pandas
- joblib
- plotly
- requests
- scikit-learn
- beautifulsoup4

### Troubleshooting
- If deployment fails, check the logs in Streamlit Cloud
- Ensure all files (models/, data/, assets/) are in the repository
- Verify requirements.txt has all dependencies

### Local Testing
Before deploying, test locally:
```bash
streamlit run app.py
```

### Support
- Streamlit Docs: https://docs.streamlit.io/
- Community Forum: https://discuss.streamlit.io/