import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import ast
import requests

class MovieRecommender:
    def __init__(self, movies_path='dataset/tmdb_5000_movies.csv', credits_path='dataset/tmdb_5000_credits.csv'):
        try:
            import os
            import pickle
            
            # Paths for cached data
            cache_dir = 'dataset/cache'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            processed_data_path = os.path.join(cache_dir, 'processed_movies.pkl')
            similarity_path = os.path.join(cache_dir, 'similarity.pkl')
            vector_path = os.path.join(cache_dir, 'vector.pkl')

            if os.path.exists(processed_data_path) and os.path.exists(similarity_path):
                print("Loading cached data...")
                with open(processed_data_path, 'rb') as f:
                    self.movies = pickle.load(f)
                with open(similarity_path, 'rb') as f:
                    self.similarity = pickle.load(f)
                if os.path.exists(vector_path):
                    with open(vector_path, 'rb') as f:
                        self.vector = pickle.load(f)
                print("Cache loaded successfully!")
                return

            print(f"Checking for raw files in: {os.getcwd()}")
            if not os.path.exists(movies_path):
                print(f"Missing: {movies_path}")
            if not os.path.exists(credits_path):
                print(f"Missing: {credits_path}")
                
            self.movies = pd.read_csv(movies_path)
            self.credits = pd.read_csv(credits_path)
            self.movies['genres_raw'] = self.movies['genres'] # Keep raw for UI
            self.prepare_data()
            
            # Save to cache
            print("Saving processed data to cache...")
            with open(processed_data_path, 'wb') as f:
                pickle.dump(self.movies, f)
            with open(similarity_path, 'wb') as f:
                pickle.dump(self.similarity, f)
            with open(vector_path, 'wb') as f:
                pickle.dump(self.vector, f)
            
            print("Successfully loaded and cached TMDB dataset!")
        except FileNotFoundError:
            print("CSV files not found. Using fallback demo data.")
            self.load_dummy_data()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.load_dummy_data()

    def convert_json(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def get_director(self, obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    def prepare_data(self):
        if self.movies.empty or self.credits.empty:
            return

        # Merge datasets
        self.movies = self.movies.merge(self.credits, on='title')

        # Select relevant columns
        self.movies = self.movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date']]
        
        # Clean data
        self.movies.dropna(inplace=True)
        
        self.movies['genres'] = self.movies['genres'].apply(self.convert_json)
        self.movies['keywords'] = self.movies['keywords'].apply(self.convert_json)
        self.movies['cast'] = self.movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
        self.movies['crew'] = self.movies['crew'].apply(self.get_director)
        
        # Remove spaces in names/genres to treat as single tags
        self.movies['genres'] = self.movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies['keywords'] = self.movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies['cast'] = self.movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies['crew'] = self.movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
        
        # Create tags soup
        self.movies['tags'] = (
            self.movies['overview'].apply(lambda x: x.split()) + 
            self.movies['genres'] + 
            self.movies['keywords'] + 
            self.movies['cast'] + 
            self.movies['crew']
        )
        self.movies['tags_string'] = self.movies['tags'].apply(lambda x: " ".join(x).lower())
        
        # Setup Vectorizer
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.vector = self.tfidf.fit_transform(self.movies['tags_string']).toarray()
        self.similarity = cosine_similarity(self.vector)

    def get_recommendations(self, movie_title):
        if self.movies.empty:
            return []
            
        try:
            # Try exact match first
            matches = self.movies[self.movies['title'].str.lower() == movie_title.lower()]
            
            if matches.empty:
                # Fuzzy match using difflib
                import difflib
                all_titles = self.movies['title'].tolist()
                close_matches = difflib.get_close_matches(movie_title, all_titles, n=1, cutoff=0.3)
                
                if close_matches:
                    target_title = close_matches[0]
                    matches = self.movies[self.movies['title'] == target_title]
                else:
                    # Try partial match
                    matches = self.movies[self.movies['title'].str.lower().str.contains(movie_title.lower())]
            
            if matches.empty:
                return []
                
            movie_index = matches.index[0]
            distances = self.similarity[movie_index]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11] # Show 10 movies
            
            recommendations = []
            for i in movies_list:
                m_idx = i[0]
                movie = self.movies.iloc[m_idx]
                
                recommendations.append({
                    'title': movie['title'],
                    'id': int(movie['movie_id']),
                    'overview': movie['overview'],
                    'genres': ", ".join(movie['genres']) if isinstance(movie['genres'], list) else str(movie['genres']),
                    'match': min(100, round(i[1] * 150)), # Boosted for better UI feedback
                    'poster_path': None
                })
            
            # Add the matched movie itself as the first element to show what was found
            matched_movie = self.movies.iloc[movie_index]
            return {
                'matched_title': matched_movie['title'],
                'recommendations': recommendations
            }
        except Exception as e:
            print(f"Error finding movie: {e}")
            return []

    def get_all_movies(self, limit=20, offset=0):
        if self.movies.empty:
            return []
        
        # Return a paginated sample of movies
        sample = self.movies.iloc[offset : offset + limit]
        results = []
        for _, movie in sample.iterrows():
            results.append({
                'title': movie['title'],
                'id': int(movie['movie_id']),
                'overview': movie['overview'],
                'genres': ", ".join(movie['genres']) if isinstance(movie['genres'], list) else str(movie['genres']),
                'release_date': str(movie['release_date']) if pd.notnull(movie['release_date']) else "Unknown",
                'poster_path': None
            })
        return results

    def search_collection(self, query):
        if self.movies.empty:
            return []
        
        query_clean = "".join(filter(str.isalnum, query.lower()))
        
        # 1. Filter using normalized contains
        self.movies['title_clean'] = self.movies['title'].str.lower().apply(lambda x: "".join(filter(str.isalnum, x)))
        matches = self.movies[self.movies['title_clean'].str.contains(query_clean)].copy()
        
        # 2. Add fuzzy matches ONLY if no partial matches were found
        if matches.empty:
            import difflib
            all_titles = self.movies['title'].tolist()
            # Increased cutoff to 0.7 for much stricter matching
            close_matches = difflib.get_close_matches(query, all_titles, n=10, cutoff=0.7)
            if close_matches:
                matches = self.movies[self.movies['title'].isin(close_matches)].copy()

        if matches.empty:
            return []
            
        # Convert release_date to datetime for sorting
        matches['release_date_dt'] = pd.to_datetime(matches['release_date'], errors='coerce')
        
        # Sort by release date descending
        matches = matches.sort_values(by='release_date_dt', ascending=False)
        
        results = []
        for _, movie in matches.iterrows():
            results.append({
                'title': movie['title'],
                'id': int(movie['movie_id']),
                'overview': movie['overview'],
                'genres': ", ".join(movie['genres']) if isinstance(movie['genres'], list) else str(movie['genres']),
                'release_date': movie['release_date'] if pd.notnull(movie['release_date']) else "Unknown",
                'poster_path': None
            })
        return results

    def get_total_count(self):
        return len(self.movies) if not self.movies.empty else 0

    def load_dummy_data(self):
        data = [
            {"movie_id": 1, "title": "Inception", "overview": "A thief who steals corporate secrets through the use of dream-sharing technology.", "genres": ["Action", "Sci-Fi"], "tags_string": "action sci-fi dream thief"},
            {"movie_id": 2, "title": "Interstellar", "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.", "genres": ["Sci-Fi", "Adventure"], "tags_string": "sci-fi adventure space wormhole"},
            {"movie_id": 3, "title": "The Matrix", "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality.", "genres": ["Action", "Sci-Fi"], "tags_string": "action sci-fi reality hacker"},
            {"movie_id": 4, "title": "The Martian", "overview": "An astronaut becomes stranded on Mars and must rely on his ingenuity to find a way to signal Earth.", "genres": ["Sci-Fi", "Drama"], "tags_string": "sci-fi drama mars astronaut"},
            {"movie_id": 5, "title": "Arrival", "overview": "A linguist works with the military to communicate with alien lifeforms after mysterious spacecraft appear.", "genres": ["Sci-Fi", "Mystery"], "tags_string": "sci-fi mystery aliens linguist"},
            {"movie_id": 6, "title": "Gravity", "overview": "Two astronauts work together to survive after an accident leaves them stranded in space.", "genres": ["Sci-Fi", "Thriller"], "tags_string": "sci-fi thriller space astronauts"},
            {"movie_id": 7, "title": "Contact", "overview": "A scientist searches for extraterrestrial intelligence and makes a profound discovery.", "genres": ["Sci-Fi", "Drama"], "tags_string": "sci-fi drama aliens scientist"}
        ]
        self.movies = pd.DataFrame(data)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.vector = self.tfidf.fit_transform(self.movies['tags_string']).toarray()
        self.similarity = cosine_similarity(self.vector)

    def get_movie_suggestions(self):
        return self.movies['title'].tolist() if not self.movies.empty else []
