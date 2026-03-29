import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load movie data
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

df = load_data()

# Convert genres to string format
df['genres'] = df['genres'].fillna('').astype(str)

# Vectorize genres
cv = CountVectorizer()
genre_matrix = cv.fit_transform(df['genres'])
cosine_sim = cosine_similarity(genre_matrix)

st.title("🎬 Genre-Based Movie Recommender")

# User selects a genre
genre_input = st.text_input("Enter a genre (e.g., Action, Comedy, Horror):")

if genre_input:
    st.subheader(f"Top 5 {genre_input} Movies")
    filtered = df[df['genres'].str.contains(genre_input, case=False, na=False)]
    if not filtered.empty:
        st.write(filtered[['title', 'genres']].head(5))
    else:
        st.warning("No matching genre found.")
