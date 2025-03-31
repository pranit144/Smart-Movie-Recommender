from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)

# ---------------------------
# Load saved model components
# ---------------------------
with open("train_df.pkl", "rb") as f:
    train_df = pickle.load(f)
with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)
with open("train_genre_features.pkl", "rb") as f:
    train_genre_features = pickle.load(f)
# (Optional) Loading nn_model if needed:
# with open("nn_model.pkl", "rb") as f:
#     nn_model = pickle.load(f)

# If train_df does not have a list of genres, create it from the 'genres' column
if "genre_list" not in train_df.columns:
    train_df["genre_list"] = train_df["genres"].apply(lambda x: x.split("|"))

# Prepare a list of all genres (for the dropdown options)
all_genres = sorted({genre for sublist in train_df["genre_list"] for genre in sublist})

# Prepare rating options (0 to 5 in increments of 0.5)
rating_options = [str(i / 2) for i in range(0, 11)]
# Prepare recommendation number options (1 to 10)
recommendation_options = [str(i) for i in range(1, 11)]


# ---------------------------
# Recommendation function using the training model
# ---------------------------
def recommend_movies_train(input_genres, min_rating, max_rating, n_recommendations=5, use_filter=True):
    """
    Recommend movies using the training set model.

    Parameters:
        input_genres (str): Comma-separated string of genres (e.g., "Comedy, Drama")
        min_rating (float): Minimum average rating.
        max_rating (float): Maximum average rating.
        n_recommendations (int): Number of recommendations.
        use_filter (bool): Whether to filter the training set by genre string matching.

    Returns:
        pd.DataFrame: Recommended movies from the training set.
    """
    # Clean and process the input genres: replace any "|" with commas and split
    cleaned_input = input_genres.replace("|", ",")
    input_genre_list = [g.strip() for g in cleaned_input.split(',') if g.strip()]

    if not input_genre_list:
        return pd.DataFrame()

    # Filter training movies by the given rating range
    filtered_train = train_df[(train_df['avg_rating'] >= min_rating) & (train_df['avg_rating'] <= max_rating)]

    # Optionally filter training movies to keep those that have one of the input genres
    if use_filter:
        genre_pattern = '|'.join(input_genre_list)
        filtered_train = filtered_train[filtered_train['genres'].str.contains(genre_pattern, case=False, na=False)]

    if filtered_train.empty:
        return pd.DataFrame()

    # Get indices of the filtered training data relative to the full training set
    filtered_indices = filtered_train.index.to_numpy()

    # Obtain the corresponding genre features from the training set features
    filtered_features = train_genre_features[[list(train_df.index).index(i) for i in filtered_indices]]

    # Create the input vector using the same MultiLabelBinarizer
    input_vector = mlb.transform([input_genre_list])

    # Fit a temporary nearest neighbors model on the filtered training data
    nn_filtered = NearestNeighbors(metric='cosine')
    nn_filtered.fit(filtered_features)

    n_neighbors = min(n_recommendations, len(filtered_train))
    distances, indices = nn_filtered.kneighbors(input_vector, n_neighbors=n_neighbors)

    # Map relative indices back to the original training DataFrame indices
    recommended_indices = filtered_indices[indices[0]]

    return train_df.loc[recommended_indices][['movieId', 'title', 'avg_rating', 'genres']]


# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form inputs from dropdowns
        selected_genres = request.form.getlist("genres")
        # Join selected genres into a comma-separated string
        input_genres = ", ".join(selected_genres)

        try:
            min_rating = float(request.form.get("min_rating", 0))
            max_rating = float(request.form.get("max_rating", 5))
            n_recommendations = int(request.form.get("n_recommendations", 5))
        except ValueError:
            return render_template("index.html", error="Invalid rating or recommendation number.",
                                   all_genres=all_genres, rating_options=rating_options,
                                   recommendation_options=recommendation_options)

        if min_rating > max_rating:
            return render_template("index.html", error="Minimum rating cannot be greater than maximum rating.",
                                   all_genres=all_genres, rating_options=rating_options,
                                   recommendation_options=recommendation_options)

        # Get recommendations from the model
        recommendations = recommend_movies_train(input_genres, min_rating, max_rating, n_recommendations)

        if recommendations.empty:
            message = "No movies found for the given criteria."
            return render_template("index.html", message=message,
                                   all_genres=all_genres, rating_options=rating_options,
                                   recommendation_options=recommendation_options)
        else:
            # Convert DataFrame to HTML table for display
            rec_html = recommendations.to_html(classes="table table-striped", index=False)
            return render_template("index.html", recommendations=rec_html,
                                   all_genres=all_genres, rating_options=rating_options,
                                   recommendation_options=recommendation_options)

    # GET request: pass dropdown options to template
    return render_template("index.html", all_genres=all_genres, rating_options=rating_options,
                           recommendation_options=recommendation_options)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()
    input_genres = data.get("genres", "")
    try:
        min_rating = float(data.get("min_rating", 0))
        max_rating = float(data.get("max_rating", 5))
        n_recommendations = int(data.get("n_recommendations", 5))
    except ValueError:
        return jsonify({"error": "Invalid rating or recommendation number."}), 400

    recommendations = recommend_movies_train(input_genres, min_rating, max_rating, n_recommendations)
    if recommendations.empty:
        return jsonify({"message": "No movies found for the given criteria."})

    result = recommendations.to_dict(orient="records")
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
