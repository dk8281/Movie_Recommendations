import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("movielens_100k_prepared.csv")

# Data cleaning
if "video_release_date" in df.columns:
    df = df.drop(columns=["video_release_date"])
df["release_date"] = df["release_date"].fillna("Unknown")
df["IMDb_URL"] = df["IMDb_URL"].fillna("Unknown")

# ---------------------------
# Collaborative Filtering Setup
# ---------------------------
user_item_matrix = df.pivot_table(index="user_id", columns="title", values="rating").fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity,
                                  index=user_item_matrix.index,
                                  columns=user_item_matrix.index)

# ---------------------------
# Content-Based Filtering Setup
# ---------------------------
genre_cols = ["unknown","Action","Adventure","Animation","Children","Comedy","Crime",
              "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
              "Romance","Sci-Fi","Thriller","War","Western"]

movie_genres = df[["title"] + genre_cols].drop_duplicates().set_index("title")
genre_similarity = cosine_similarity(movie_genres)
genre_similarity_df = pd.DataFrame(genre_similarity,
                                   index=movie_genres.index,
                                   columns=movie_genres.index)

# ---------------------------
# Hybrid Recommendation Function
# ---------------------------
def hybrid_recommend(user_id, liked_movie, n=5, alpha=0.5):
    if user_id not in user_item_matrix.index:
        return [f"⚠️ User {user_id} not found."]
    if liked_movie not in genre_similarity_df:
        return [f"⚠️ Movie '{liked_movie}' not found."]

    # Collaborative Filtering
    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).drop(user_id)
    weighted_ratings = np.dot(similar_users, user_item_matrix.loc[similar_users.index]) / similar_users.sum()
    collaborative_scores = pd.Series(weighted_ratings, index=user_item_matrix.columns)

    # Content-Based Filtering
    content_scores = genre_similarity_df[liked_movie]

    # Hybrid Score
    hybrid_scores = alpha * collaborative_scores + (1 - alpha) * content_scores

    # Exclude already rated
    user_rated = user_item_matrix.loc[user_id]
    hybrid_scores = hybrid_scores[user_rated[user_rated == 0].index]

    return hybrid_scores.sort_values(ascending=False)[:n].index.tolist()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Hybrid Movie Recommendation System")
st.sidebar.header("🔧 User Preferences")

user_id = st.sidebar.number_input("Enter User ID:", 
                                  min_value=int(df["user_id"].min()), 
                                  max_value=int(df["user_id"].max()), 
                                  value=10)

movie_choice = st.sidebar.selectbox("Select a Movie you liked:", sorted(df["title"].unique()))

alpha = st.sidebar.slider("⚖️ Preference Weight (Collaborative vs Content-Based)", 0.0, 1.0, 0.5)

n_recs = st.sidebar.slider("📌 Number of Recommendations", 1, 10, 5)

if st.sidebar.button("Get Recommendations"):
    recs = hybrid_recommend(user_id, movie_choice, n=n_recs, alpha=alpha)
    st.subheader(f"🎥 Top {n_recs} Recommendations for User {user_id}")
    for i, movie in enumerate(recs, 1):
        st.write(f"{i}. {movie}")
