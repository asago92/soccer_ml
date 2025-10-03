import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
#import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Soccer Match Predictor", layout="wide")

# -------------------------
# Load or Create Sample Data
# -------------------------
def create_sample_data():
    data = {
        "home_team": ["Team A", "Team B", "Team C", "Team A", "Team D", "Team B", "Team C", "Team D"],
        "away_team": ["Team B", "Team A", "Team D", "Team C", "Team A", "Team C", "Team B", "Team A"],
        "home_goals_scored": [2, 1, 3, 0, 2, 1, 2, 0],
        "away_goals_scored": [1, 2, 0, 2, 1, 2, 1, 3],
        "home_form": [3, 2, 4, 1, 3, 2, 3, 1],  # points in last 5 games
        "away_form": [2, 3, 1, 2, 2, 3, 2, 4],
        "result": ["Win", "Lose", "Win", "Lose", "Win", "Draw", "Win", "Lose"]
    }
    return pd.DataFrame(data)

# -------------------------
# Train or Load Model
# -------------------------
def train_model(df):
    le_result = LabelEncoder()
    df["target"] = le_result.fit_transform(df["result"])

    X = df[["home_goals_scored", "away_goals_scored", "home_form", "away_form"]]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_result

# Load or Train model
if os.path.exists("soccer_model.pkl"):
    model = joblib.load("soccer_model.pkl")
    le_result = joblib.load("label_encoder.pkl")
    df = pd.read_csv("soccer_data.csv")
else:
    df = create_sample_data()
    model, le_result = train_model(df)
    joblib.dump(model, "soccer_model.pkl")
    joblib.dump(le_result, "label_encoder.pkl")
    df.to_csv("soccer_data.csv", index=False)

# -------------------------
# Streamlit UI
# -------------------------
st.title("⚽ Soccer Match Outcome Predictor")
st.write("Predict whether a team will **Win, Lose, or Draw** based on historical data.")

teams = list(set(df["home_team"].unique()).union(set(df["away_team"].unique())))
col1, col2 = st.columns(2)
home_team = col1.selectbox("Select Home Team", teams)
away_team = col2.selectbox("Select Away Team", [t for t in teams if t != home_team])

# Sample feature engineering (replace with real stats in production)
home_goals = df[df["home_team"] == home_team]["home_goals_scored"].mean()
away_goals = df[df["away_team"] == away_team]["away_goals_scored"].mean()
home_form = df[df["home_team"] == home_team]["home_form"].mean()
away_form = df[df["away_team"] == away_team]["away_form"].mean()

# Handle NaN if no data
if np.isnan(home_goals): home_goals = 1.5
if np.isnan(away_goals): away_goals = 1.2
if np.isnan(home_form): home_form = 2
if np.isnan(away_form): away_form = 2

features = np.array([[home_goals, away_goals, home_form, away_form]])
prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]

# Output
st.subheader("📊 Prediction Result")
result_label = le_result.inverse_transform([prediction])[0]
st.write(f"**Predicted Outcome:** {home_team} will **{result_label}**")

st.subheader("🔢 Probabilities")
prob_df = pd.DataFrame({
    "Outcome": le_result.inverse_transform([0,1,2]),
    "Probability (%)": np.round(probabilities * 100, 2)
})
st.dataframe(prob_df, use_container_width=True)

st.write(f"✅ Confidence Level: **{np.max(probabilities)*100:.2f}%**")

# -------------------------
# Show Tree Votes (Stacked Horizontal Bar)
# -------------------------
st.subheader("🌳 How the Forest Voted")

# Get predictions from each tree in the forest
tree_votes = [tree.predict(features)[0] for tree in model.estimators_]

# Count how many trees voted for each class
vote_counts = pd.Series(tree_votes).value_counts()
vote_counts = vote_counts.reindex([0,1,2], fill_value=0)  # Ensure all outcomes exist
vote_counts.index = le_result.inverse_transform(vote_counts.index)

# Identify predicted outcome
predicted_outcome = le_result.inverse_transform([prediction])[0]

# Colors for stacked bar
colors_map = {
    "Win": "#2ecc71",   # green
    "Draw": "#f39c12",  # orange
    "Lose": "#e74c3c"   # red
}
colors = [colors_map[outcome] for outcome in vote_counts.index]

# Plot stacked horizontal bar
fig, ax = plt.subplots(figsize=(7,2))
ax.barh(["Votes"], vote_counts.values, color=colors, stacked=True)

# Add text labels inside each segment
for i, (outcome, count) in enumerate(zip(vote_counts.index, vote_counts.values)):
    if count > 0:
        ax.text(sum(vote_counts.values[:i]) + count/2, 0, 
                f"{outcome}: {count}", 
                ha="center", va="center", color="white", fontweight="bold")

ax.set_xlim(0, sum(vote_counts.values))
ax.set_xlabel("Number of Trees Voting")
ax.set_title("Distribution of Tree Votes (Stacked Bar)")

st.pyplot(fig)

# -------------------------
# Allow User to Add New Data
# -------------------------
st.subheader("📥 Update Model with New Match Data")
with st.form("update_form"):
    ht = st.selectbox("Home Team", teams, key="ht")
    at = st.selectbox("Away Team", [t for t in teams if t != ht], key="at")
    h_goals = st.number_input("Home Goals", min_value=0, max_value=10, value=1)
    a_goals = st.number_input("Away Goals", min_value=0, max_value=10, value=1)
    h_form = st.slider("Home Team Form (last 5 games)", 0, 5, 3)
    a_form = st.slider("Away Team Form (last 5 games)", 0, 5, 2)
    res = st.selectbox("Match Result", ["Win", "Lose", "Draw"])
    submitted = st.form_submit_button("Update Model")

    if submitted:
        new_row = {
            "home_team": ht,
            "away_team": at,
            "home_goals_scored": h_goals,
            "away_goals_scored": a_goals,
            "home_form": h_form,
            "away_form": a_form,
            "result": res
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv("soccer_data.csv", index=False)
        model, le_result = train_model(df)
        joblib.dump(model, "soccer_model.pkl")
        joblib.dump(le_result, "label_encoder.pkl")
        st.success("✅ Model retrained with new data!")
