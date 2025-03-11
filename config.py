import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_PATH: str = "data"
EXTERNAL_DATA_PATH: str = "external_data"
MODELS_PATH: str = "models"

DEFAULT_SCHEMA = {
    "title": str,
    "release_year": int,
    "box_office": float,
    "budget": float,
    "rating": str,
    "collection": str,
    "cast": list[str],
    "director": str,
    "writers": list[str],
    "distributors": list[str],
    "genres": list[str],
    "plot": str
}

DEFAULT_SCHMA_MAPPING = {
    "writer": "writers",
    "distributor": "distributors",
    "genre": "genres",
    "belongs_to_collection": "collection",
}