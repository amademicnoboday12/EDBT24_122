import os

from .flight import load_flight
from .book import load_book
from .lastfm import load_lastfm
from .expedia import load_expedia
from .movie import load_movie
from .walmart import load_walmart
from .yelp import load_yelp

data_root = os.environ.get("DATA_ROOT", "data/Hamlet/")

def load_dataset(dataset: str, data_folder=None):
    if dataset == "flight":
        return load_flight(data_folder or f"{data_root}/Flights")
    elif dataset == "book":
        return load_book(data_folder or f"{data_root}/BookCrossing")
    elif dataset == "lastfm":
        return load_lastfm(data_folder or f"{data_root}/Flights")
    elif dataset == "expedia":
        return load_expedia(data_folder or f"{data_root}/Expedia")
    elif dataset == "movie":
        return load_movie(data_folder or f"{data_root}/MovieLens1M")
    elif dataset == "walmart":
        return load_walmart(data_folder or f"{data_root}/Walmart")
    elif dataset == "yelp":
        return load_yelp(data_folder or f"{data_root}/Yelp")
    else:
        raise ValueError("Not implemented yet")
