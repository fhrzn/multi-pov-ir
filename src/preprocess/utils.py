import unicodedata
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import plotly.colors as pc
import plotly.graph_objects as go
from wordcloud import WordCloud

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
                else np.array(vectorizer.transform([e for e in ents if e is not None]).mean(axis=0))
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


def plot_pov_poi(
    spot_df,
    entity_df,
    center: Dict[str, float] = {"lat": 35.6895, "lon": 139.7517},
    zoom: int = 12,
    show_all_poi: bool = False
):
    spot_pd = spot_df.to_pandas()
    entity_pd = entity_df.to_pandas()
    if show_all_poi is False:
        entity_pd = entity_df.filter(
            entity_df["nearby_pov_cluster"].list.len() > 0
        ).to_pandas()

    palette = pc.qualitative.Plotly
    unique_clusters = sorted(c for c in spot_pd["cluster_id"].unique() if c != -1)
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}
    color_map[-1] = "lightgray"
    pov_colors = spot_pd["cluster_id"].map(color_map).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=spot_pd["latitude"],
            lon=spot_pd["longitude"],
            mode="markers",
            marker=dict(size=6, color=pov_colors, opacity=0.7),
            name="POV",
            hovertemplate="POV<br>Cluster ID: %{customdata[0]}<extra></extra>",
            customdata=spot_pd[["cluster_id"]].values,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=entity_pd["latitude"],
            lon=entity_pd["longitude"],
            mode="markers",
            marker=dict(size=10, color="#2893c1", opacity=0.7),
            name="POI",
            hovertemplate="POI %{customdata[0]} (%{customdata[1]})<br>Nearby Cluster: [%{customdata[2]}]<extra></extra>",
            customdata=entity_pd[["entity_id", "poi_name", "nearby_pov_cluster"]].values,
        )
    )
    fig.update_layout(
        map=dict(center=center, zoom=zoom),
        height=800,
        width=1200,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(x=0, y=1),
    )
    return fig


def plot_wordcloud(df, id_col: str, id_val: int, text_col: str, title: str = None, font_path: str = None):
    row = df.filter(df[id_col] == id_val).to_dicts()
    if not row:
        raise ValueError(f"{id_col}={id_val} not found")
    tokens = [e for e in row[0][text_col] if e] if row[0][text_col] else []
    text = " ".join(tokens) or "unknown"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        WordCloud(width=800, height=800, background_color="white", font_path=font_path).generate(text),
        interpolation="bilinear",
    )
    ax.set_title(title or f"{id_col}={id_val}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plot_entity_wordcloud(entity_df, cluster_df, n_samples: int = 3):
    import polars as pl

    cluster_entities = {
        row["cluster_id"]: [e for e in row["entities"] if e]
        for row in cluster_df.to_dicts()
    }

    candidates = entity_df.filter(
        pl.col("nearby_pov_cluster").list.len() > 0
    ).sample(n=n_samples, seed=42).to_dicts()

    fig, ax = plt.subplots(n_samples, 2, figsize=(10, n_samples * 3.5))
    if n_samples == 1:
        ax = [ax]

    for i, row in enumerate(candidates):
        pov_entities = [
            e
            for cid in row["nearby_pov_cluster"]
            for e in cluster_entities.get(cid, [])
        ]

        for j, (tokens, title) in enumerate([
            (row["poi_name_tags"], row["poi_name"]),
            (pov_entities, "Nearby Cluster NER"),
        ]):
            text = " ".join([t for t in tokens if t]) or "unknown"
            ax[i][j].imshow(
                WordCloud(width=800, height=800, background_color="white").generate(text),
                interpolation="bilinear",
            )
            ax[i][j].set_title(title)

    for a in np.array(ax).flat:
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    return fig


def normalize_text(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()
