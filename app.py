import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_data
from recommender import recommendation_system
from recommender import get_recommendations


st.set_page_config(
    page_title="Netflix Analytics",
    layout="wide"
)

st.title("Netflix Analytics Dashboard")

# Load dataset
df = load_data()

st.sidebar.title("Navigation")

option = st.sidebar.radio(
    "Select Option",
    [
        "Dataset Overview",
        "Movies vs TV Shows",
        "Top Genres",
        "Country Analysis",
        "Recommendation System"
    ]
)

# Dataset Overview
if option == "Dataset Overview":

    st.subheader("Netflix Dataset")

    st.write(df.head())

    st.write("Shape of Dataset:", df.shape)

# Movies vs TV Shows
elif option == "Movies vs TV Shows":

    st.subheader("Movies vs TV Shows")

    type_count = df['type'].value_counts()

    fig, ax = plt.subplots()

    ax.pie(
        type_count,
        labels=type_count.index,
        autopct='%1.1f%%'
    )

    st.pyplot(fig)

# Top Genres
elif option == "Top Genres":

    st.subheader("Top Genres")

    genres = df['listed_in'].str.split(
        ', ', expand=True
    ).stack()

    top_genres = genres.value_counts().head(10)

    fig, ax = plt.subplots()

    top_genres.plot(
        kind='bar',
        ax=ax
    )

    st.pyplot(fig)

# Country Analysis
elif option == "Country Analysis":

    st.subheader("Top Countries")

    countries = df['country'].str.split(
        ', ', expand=True
    ).stack()

    top_country = countries.value_counts().head(10)

    fig, ax = plt.subplots()

    top_country.plot(
        kind='bar',
        ax=ax
    )

    st.pyplot(fig)

# Recommendation System
elif option == "Recommendation System":

    st.subheader("Netflix Recommendation")

    cosine_sim, indices = recommendation_system(df)

    movie_list = df['title'].unique()

    selected_movie = st.selectbox(
        "Choose Movie",
        movie_list
    )

    if st.button("Recommend"):

        recommendations = get_recommendations(
            selected_movie,
            cosine_sim,
            indices,
            df
        )

        st.write(
            "Recommended Movies:"
        )

        for movie in recommendations:
            st.write(movie)
