from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from recommender import MovieRecommender
import os
import requests
import uvicorn
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Movie Recommender API")

# Mount static files
if not os.path.exists('static'):
    os.makedirs('static')
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 environment manually to avoid compatibility issues with Python 3.14 + Starlette templates
if not os.path.exists('templates'):
    os.makedirs('templates')
jinja_env = Environment(loader=FileSystemLoader("templates"))

# TMDB API Configuration
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Simple in-memory cache for posters to speed up loading
poster_cache = {}

def get_poster_path(movie_id):
    if movie_id in poster_cache:
        return poster_cache[movie_id]
        
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=2)
        data = response.json()
        path = TMDB_IMAGE_BASE + data.get('poster_path') if data.get('poster_path') else None
        poster_cache[movie_id] = path
        return path
    except:
        return None

# Initialize recommender
recommender = MovieRecommender()

class RecommendRequest(BaseModel):
    movie_title: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        template = jinja_env.get_template("index.html")
        content = template.render(request=request)
        return HTMLResponse(content=content)
    except Exception as e:
        print(f"Template Error: {e}")
        raise HTTPException(status_code=500, detail=f"Template Error: {str(e)}")

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    movie_title = req.movie_title
    
    if not movie_title:
        raise HTTPException(status_code=400, detail="No movie title provided")
        
    try:
        data = recommender.get_recommendations(movie_title)
        
        if not data:
            raise HTTPException(status_code=404, detail="Movie not found or no recommendations available")
            
        recommendations = data['recommendations']
        matched_title = data['matched_title']
            
        # Enrich with posters
        for movie in recommendations:
            movie['poster_url'] = get_poster_path(movie['id'])
            
        return {
            'original_movie': movie_title,
            'matched_movie': matched_title,
            'recommendations': recommendations
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/movies")
async def get_movies(limit: int = 20, offset: int = 0):
    try:
        total = recommender.get_total_count()
        movies = recommender.get_all_movies(limit=limit, offset=offset)
        for movie in movies:
            movie['poster_url'] = get_poster_path(movie['id'])
        return {
            "movies": movies,
            "total": total,
            "offset": offset,
            "limit": limit
        }
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/search_collection")
async def search_collection(q: str = ""):
    try:
        if not q:
            return {"movies": []}
            
        movies = recommender.search_collection(q)
        for movie in movies:
            movie['poster_url'] = get_poster_path(movie['id'])
            
        return {
            "movies": movies,
            "total": len(movies)
        }
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/suggestions")
async def suggestions(q: str = ""):
    all_titles = recommender.get_movie_suggestions()
    
    if not q:
        return all_titles[:10]
        
    filtered = [t for t in all_titles if q.lower() in t.lower()][:10]
    return filtered

if __name__ == "__main__":
    print("\n" + "="*40)
    print("SERVER STARTING ON http://localhost:8000")
    print("="*40 + "\n")
    uvicorn.run(app, host="localhost", port=8000)
