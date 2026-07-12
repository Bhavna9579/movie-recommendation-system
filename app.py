import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- Load CSV Data --------------------

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# -------------------- Merge Data --------------------

movies = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)


# Keep required columns

movies = movies[
    [
        "movie_id",
        "title_x",
        "genres",
        "keywords",
        "overview",
        "cast",
        "crew"
    ]
]


movies.rename(
    columns={"title_x": "title"},
    inplace=True
)


# -------------------- Convert JSON String --------------------

def convert(obj):
    L = []

    for i in ast.literal_eval(obj):
        L.append(i["name"])

    return L


# Genres

movies["genres"] = movies["genres"].apply(convert)

# Keywords

movies["keywords"] = movies["keywords"].apply(convert)



# -------------------- Extract Cast --------------------

def convert_cast(obj):
    L = []

    counter = 0

    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i["name"])
            counter += 1
        else:
            break

    return L


movies["cast"] = movies["cast"].apply(convert_cast)



# -------------------- Extract Director --------------------

def fetch_director(obj):

    L = []

    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])

    return L


movies["crew"] = movies["crew"].apply(fetch_director)



# -------------------- Create Tags --------------------

movies["overview"] = movies["overview"].fillna("")


movies["tags"] = (
    movies["overview"]
    + " "
    + movies["genres"].apply(lambda x: " ".join(x))
    + " "
    + movies["keywords"].apply(lambda x: " ".join(x))
    + " "
    + movies["cast"].apply(lambda x: " ".join(x))
    + " "
    + movies["crew"].apply(lambda x: " ".join(x))
)


movies["tags"] = movies["tags"].str.lower()



# -------------------- Similarity Calculation --------------------
@st.cache_data
def create_similarity(tags):

    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = cv.fit_transform(tags)

    similarity = cosine_similarity(vectors)

    return similarity


similarity = create_similarity(
    movies["tags"]
)


# -------------------- Fetch Poster --------------------

def fetch_poster(movie_id):

    api_key = "8265bd1679663a7ea12ac168da84d2e8"

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"


    try:

        response = requests.get(url)

        data = response.json()

        poster_path = data.get("poster_path")


        if poster_path:
            return (
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )

        else:
            return None


    except:

        return None



# -------------------- Recommendation --------------------

def recommend(movie):

    movie_index = movies[
        movies["title"] == movie
    ].index[0]


    distances = similarity[movie_index]


    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x:x[1]
    )[1:6]


    names = []
    posters = []


    for i in movie_list:

        movie_id = movies.iloc[i[0]].movie_id

        names.append(
            movies.iloc[i[0]].title
        )

        posters.append(
            fetch_poster(movie_id)
        )


    return names, posters



# -------------------- Streamlit UI --------------------

st.set_page_config(
    page_title="Movie Recommender",
    layout="wide"
)


st.title("🎬 Movie Recommendation System")


selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)



if st.button("Recommend"):


    names, posters = recommend(
        selected_movie
    )


    cols = st.columns(5)


    for i in range(5):

        with cols[i]:

            st.write(names[i])


            if posters[i]:

                st.image(
                    posters[i],
                    use_container_width=True
                )

            else:

                st.write(
                    "Poster not available"
                )