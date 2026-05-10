# 🎬 Movie Recommender

A high-performance, content-based movie recommendation engine built with **Python**, **FastAPI**, and **Scikit-learn**. This system analyzes metadata from 5,000+ movies to find mathematical similarities and suggest your next favorite film.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688.svg)

## ✨ Key Features

- **🚀 Lightning Fast Startup**: Implements **Pickle-based caching** for processed datasets and similarity matrices. Subsequent loads take milliseconds instead of seconds.
- **🔍 Intelligent Search**: Uses a combination of **TF-IDF Vectorization** and **Cosine Similarity** for recommendations, plus a fuzzy-matching system for the library browser.
- **📚 Collection Browser**: Explore the entire TMDB dataset with pagination and real-time title searching.
- **🎨 Premium Interface**: A modern, responsive web UI with smooth animations, dark-themed elements, and dynamic loading states.
- **🖼️ Real-time Metadata**: Fetches movie posters and details directly from the TMDB API for a rich visual experience.

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.14+)
- **Machine Learning**: Scikit-learn (TfidfVectorizer, Cosine Similarity)
- **Data Handling**: Pandas, Pickle (Caching)
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), JavaScript
- **Icons**: Lucide Icons

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/3bdelrahmanashraf/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Dataset
The app includes the processed TMDB 5000 dataset in the `dataset/` folder. On the first run, the system will process the raw CSVs and generate a cache for future use.

### 4. Run the Application
```bash
python app.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

## 🧠 How It Works

1. **Feature Extraction**: The system combines movie overviews, genres, keywords, cast, and crew into a "tags" soup.
2. **Vectorization**: Uses `TfidfVectorizer` to convert text data into numerical vectors, removing common English stop words.
3. **Similarity Scoring**: Calculates the **Cosine Similarity** between all movie vectors.
4. **Caching**: Saves the processed dataframe and similarity matrix to `dataset/cache/` to avoid re-computation on every restart.

---
