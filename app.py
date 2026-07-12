import pickle
import pandas as pd
import streamlit as st
import requests

# -------------------- Load Data --------------------

movies_dict = pickle.load(open("model/movie_list.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open("model/similarity.pkl", "rb"))

# -------------------- Fetch Poster --------------------

def fetch_poster(movie_id):
    api_key = "8265bd1679663a7ea12ac168da84d2e8"   # <-- Replace with your own TMDB API key

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

    try:
        response = requests.get(url)

        print("Movie ID:", movie_id)
        print("Status Code:", response.status_code)

        if response.status_code != 200:
            print(response.text)
            return None

        data = response.json()
        print(data)

        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            print("Poster path not found.")
            return None

    except Exception as e:
        print("Error:", e)
        return None


# -------------------- Recommend --------------------

def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    names = []
    posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]]["movie_id"]

        names.append(movies.iloc[i[0]]["title"])
        posters.append(fetch_poster(movie_id))

    return names, posters


# -------------------- Streamlit --------------------

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

if st.button("Recommend"):

    names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.write(names[i])

            if posters[i]:
                st.image(posters[i], use_container_width=True)
            else:
                st.write("❌ Poster not available")