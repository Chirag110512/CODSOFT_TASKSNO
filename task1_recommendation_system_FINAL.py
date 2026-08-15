"""
CODSOFT Internship - Task 1
Movie Recommendation System


"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. SAMPLE DATA
movies_data = [
    {"movie_id": 1, "title": "3 Idiots",           "genres": "Drama and Social Realism"},
    {"movie_id": 2, "title": "John Wick",          "genres": "Action Thriller Crime"},
    {"movie_id": 3, "title": "KGF Chapter1",        "genres": "Action Sci-Fi Thriller"},
    {"movie_id": 4, "title": "Pathaan",            "genres": "Action Thriller"},
    {"movie_id": 5, "title": "Gangs of Wasseypur ", "genres": "Crime and Underworld"},
    {"movie_id": 6, "title": "Titanic",            "genres": "Romance Drama Disaster"},
    {"movie_id": 7, "title": "Interstellar",       "genres": "Sci-Fi Drama Adventure"},
    {"movie_id": 8, "title": "The Conjuring",      "genres": "Horror Thriller Mystery"},
    {"movie_id": 9, "title": "Stree",              "genres": "Horror and Mythological fantasy"},
    {"movie_id": 10, "title": "Toy Story",         "genres": "Animation Comedy Family"},
    {"movie_id": 11, "title": "Dilwale Dulhania Le Jayenge ","genres": "Romance / Masala"},
    {"movie_id": 12, "title": "12th Fail ",        "genres": "Drama and Social Realism"},
]
movies_df = pd.DataFrame(movies_data)

# user_id -> {movie_id: rating from (1-5)}
ratings_data = [
    {"user_id": 1, "movie_id": 1, "rating": 5},
    {"user_id": 1, "movie_id": 2, "rating": 4},
    {"user_id": 1, "movie_id": 3, "rating": 5},
    {"user_id": 1, "movie_id": 12, "rating": 4},
    {"user_id": 2, "movie_id": 4, "rating": 5},
    {"user_id": 2, "movie_id": 5, "rating": 4},
    {"user_id": 2, "movie_id": 6, "rating": 5},
    {"user_id": 3, "movie_id": 1, "rating": 4},
    {"user_id": 3, "movie_id": 3, "rating": 5},
    {"user_id": 3, "movie_id": 7, "rating": 5},
    {"user_id": 3, "movie_id": 12, "rating": 5},
    {"user_id": 4, "movie_id": 8, "rating": 5},
    {"user_id": 4, "movie_id": 9, "rating": 4},
    {"user_id": 5, "movie_id": 10, "rating": 5},
    {"user_id": 5, "movie_id": 11, "rating": 5},
    {"user_id": 6, "movie_id": 1, "rating": 5},
    {"user_id": 6, "movie_id": 7, "rating": 4},
    {"user_id": 6, "movie_id": 12, "rating": 5},
]
ratings_df = pd.DataFrame(ratings_data)


# 2. CONTENT-BASED FILTERING

def build_content_similarity(movies_df):
    tfidf = TfidfVectorizer(token_pattern=r"[a-zA-Z\-]+")
    genre_matrix = tfidf.fit_transform(movies_df["genres"])
    similarity = cosine_similarity(genre_matrix)
    return similarity


def recommend_content_based(movie_title, movies_df, similarity_matrix, top_n=5):
    if movie_title not in movies_df["title"].values:
        return f"Movie '{movie_title}' not found in dataset."

    idx = movies_df.index[movies_df["title"] == movie_title][0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]  

    return movies_df.iloc[[i for i, _ in scores]][["title", "genres"]]

# 3. COLLABORATIVE FILTERING (user-based)
def build_user_item_matrix(ratings_df):
    return ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)


def recommend_collaborative(user_id, user_item_matrix, movies_df, top_n=5):
    if user_id not in user_item_matrix.index:
        return f"User {user_id} not found."

    similarity = cosine_similarity(user_item_matrix)
    sim_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # most similar users (not include me)
    similar_users = sim_df[user_id].drop(user_id).sort_values(ascending=False)

    weighted_scores = pd.Series(dtype=float)
    for other_user, sim_score in similar_users.items():
        if sim_score <= 0:
            continue
        other_ratings = user_item_matrix.loc[other_user]
        weighted_scores = weighted_scores.add(other_ratings * sim_score, fill_value=0)

    # remove movies 
    already_rated = user_item_matrix.loc[user_id]
    already_rated = already_rated[already_rated > 0].index
    weighted_scores = weighted_scores.drop(labels=already_rated, errors="ignore")

    top_movie_ids = weighted_scores.sort_values(ascending=False).head(top_n).index
    return movies_df[movies_df["movie_id"].isin(top_movie_ids)][["title", "genres"]]

# 4. DEMO
if __name__ == "__main__":
    print("=" * 60)
    print("CONTENT-BASED RECOMMENDATIONS")
    print("=" * 60)
    similarity_matrix = build_content_similarity(movies_df)
    for movie in ["3 Idiots", "Pathaan"]:
        print(f"\nBecause you liked '{movie}':")
        print(recommend_content_based(movie, movies_df, similarity_matrix, top_n=3))

    print("\n" + "=" * 60)
    print("COLLABORATIVE FILTERING RECOMMENDATIONS")
    print("=" * 60)
    user_item_matrix = build_user_item_matrix(ratings_df)
    for user_id in [1, 4]:
        print(f"\nRecommended for User {user_id}:")
        print(recommend_collaborative(user_id, user_item_matrix, movies_df, top_n=3))
