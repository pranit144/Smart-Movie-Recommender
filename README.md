# Movie Recommendation System

This is a Flask-based web application that provides movie recommendations based on genre and rating preferences. It utilizes a trained Nearest Neighbors model to find the most relevant movies based on user-selected genres and rating range.

## Features
- Users can select genres and set a rating range to receive movie recommendations.
- The system uses a pre-trained Nearest Neighbors model to find similar movies.
- API endpoint available for programmatic access to recommendations.
- Deployed on Hugging Face Spaces: [Live Demo](https://pranit144-rutu1.hf.space)

## Requirements

To run this project locally, install the required dependencies:
```bash
pip install Flask pandas scikit-learn numpy
```

## Project Structure
```
├── app.py                  # Main Flask application
├── templates/
│   ├── index.html          # Frontend UI
├── static/
│   ├── style.css           # Styling for the UI
├── train_df.pkl            # Pickled training dataset
├── mlb.pkl                 # Pickled MultiLabelBinarizer for genres
├── train_genre_features.pkl # Pickled genre feature matrix
├── README.md               # Project documentation
```

## Running the Application

1. Clone the repository and navigate into the project directory:
```bash
git clone <repo_url>
cd <repo_name>
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Flask app:
```bash
python app.py
```
4. Open a browser and go to `http://127.0.0.1:5000/` to access the UI.

## API Usage

### Endpoint: `/api/recommend`
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Request Body:**
```json
{
  "genres": "Comedy, Drama",
  "min_rating": 3.0,
  "max_rating": 5.0,
  "n_recommendations": 5
}
```
- **Response:**
```json
[
  {"movieId": 1, "title": "Movie A", "avg_rating": 4.2, "genres": "Comedy|Drama"},
  {"movieId": 2, "title": "Movie B", "avg_rating": 3.8, "genres": "Comedy"}
]
```

## Deployment
The app is deployed on Hugging Face Spaces. You can modify and redeploy it using Hugging Face or other cloud services like Vercel or AWS.

## Contributing
Feel free to fork the repository and submit pull requests. If you find any issues, open an issue in the GitHub repository.

## License
This project is licensed under the MIT License.
