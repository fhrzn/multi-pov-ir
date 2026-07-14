import unicodedata
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

R = 6371.0


def extract_entities(texts: List[str], ner_labels: List[str], model, threshold: float = 0.5) -> List[List[str]]:
    extracted = []
    for txt in tqdm(texts, desc="NER"):
        entities = model.predict_entities(txt, ner_labels, threshold=threshold)
        extracted.append([normalize_text(e["text"]) for e in entities])
    return extracted


def spatial_radius_query(
    query_coords: np.ndarray,
    index_coords: np.ndarray,
    radius_km: float,
) -> Tuple[np.ndarray, np.ndarray]:
    index_rad = np.radians(index_coords)
    query_rad = np.radians(query_coords)
    tree = BallTree(index_rad, metric="haversine")
    radius_rad = radius_km / R
    return tree.query_radius(query_rad, r=radius_rad, return_distance=True)


def compute_entity_feat(entity_lists: List[List[str]], vectorizer) -> np.ndarray:
    n_features = len(vectorizer.get_feature_names_out())
    return normalize(
        np.vstack(
            [
                np.zeros((1, n_features))
                if not ents
                else np.array(vectorizer.transform(ents).mean(axis=0))
                for ents in entity_lists
            ]
        )
    )


def spherical_centroid(lats, lons):
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    x_mean = x.mean()
    y_mean = y.mean()
    z_mean = z.mean()

    lon_centroid = np.arctan2(y_mean, x_mean)
    hyp = np.sqrt(x_mean**2 + y_mean**2)
    lat_centroid = np.arctan2(z_mean, hyp)

    return np.degrees(lat_centroid), np.degrees(lon_centroid)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def normalize_text(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()
