# Movie Recommendation System 🎬

An AI-powered content-based filtering system built with Python, Flask, and Scikit-learn.

## Features
- **TF-IDF Vectorization**: Converts movie metadata into numerical vectors.
- **Cosine Similarity**: Calculates mathematical distance to find the closest matches.
- **Premium UI**: Modern, responsive design matching professional streaming platforms.
- **Top 5 Suggestions**: Instant recommendations based on movie features (genres, cast, keywords).

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add the Dataset** (Optional but Recommended):
   - Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle.
   - Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the `dataset/` folder.
   - *Note: If no CSVs are found, the app will run in "Demo Mode" with a small sample of sci-fi movies.*

3. **Run the App**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:8000` in your browser.

## Technologies Used
- **Backend**: Python (Flask)
- **Data Science**: Pandas, Scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript (Lucide Icons)
