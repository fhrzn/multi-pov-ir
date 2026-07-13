from argparse import ArgumentParser
from typing import List
from unicodedata import normalize

import numpy as np
import polars as pl
from gliner import GLiNER
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from src.utils import get_device, normalize_text, spherical_centroid

R = 6371.0
NER_MODEL_NAME = "urchade/gliner_multi-v2.1"


def assign_cluster(df: pl.DataFrame):
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())

    eps_km = 0.1
    eps_rad = eps_km / R

    dbscan = DBSCAN(eps=eps_rad, metric="haversine", min_samples=50)
    labels = dbscan.fit_predict(coords)
    df = (
        df.with_columns(pl.Series("cluster_id", labels))
        .filter(pl.col("cluster_id") > -1)
        .with_row_index("row_idx")
    )

    return df


def assign_cluster_neighbor(df: pl.DataFrame):
    # get cluster centroid
    df_centroid = []
    for cluster_id, item in df.group_by("cluster_id"):
        latlons = item[["latitude", "longitude"]]
        lats = latlons[:, 0]
        lons = latlons[:, 1]

        clat, clon = spherical_centroid(lats, lons)
        df_centroid.append(
            {
                "cluster_id": cluster_id[0],
                "latitude": clat.item(),
                "longitude": clon.item(),
                "n_points": len(item),
            }
        )

    # identify cluster members
    df_centroid = pl.DataFrame(df_centroid).sort("cluster_id")
    df_centroid = df_centroid.join(
        df.group_by("cluster_id").agg(pl.col("row_idx").alias("member")),
        on="cluster_id",
    ).sort("cluster_id")

    # identify cluster neighbors
    centroid_coords = df_centroid[["latitude", "longitude"]].to_numpy()
    centroid_coords = np.radians(centroid_coords)

    tree = BallTree(centroid_coords, metric="haversine")
    query_rads = 1 / R
    indices, distances = tree.query_radius(
        centroid_coords, r=query_rads, return_distance=True
    )

    cluster_neighbor = []
    for i, (ind, dist) in enumerate(zip(indices, distances)):
        cluster_neighbor.append(
            {"cluster_id": i, "neighbors": ind.tolist(), "distances": dist.tolist()}
        )
    cluster_neighbor = pl.DataFrame(cluster_neighbor)

    df_centroid = df_centroid.join(cluster_neighbor, on="cluster_id")

    return df_centroid


def merge_cluster(
    pov_df: pl.DataFrame,
    centroid_df: pl.DataFrame,
    ner_labels: List[str],
    threshold: float = 0.5,
):
    # extract entities
    model = GLiNER.from_pretrained(NER_MODEL_NAME, load_tokenizer=True).to(get_device())

    extracted_ner = []
    for txt in tqdm(pov_df["text"].to_list(), desc="NER"):
        entities = model.predict_entities(txt, ner_labels, threshold=threshold)
        extracted_ner.append([normalize_text(e["text"]) for e in entities])

    pov_df = pov_df.with_columns(pl.Series("entities", extracted_ner))
    centroid_df = centroid_df.join(
        pov_df.group_by("cluster_id").agg(
            pl.col("entities").list.explode(keep_nulls=False, empty_as_null=False)
        ),
        on="cluster_id",
        how="left",
    )

    # similarity score
    all_centroids = centroid_df.to_dicts()
    ## vectorize
    vectorizer = TfidfVectorizer()
    vectorizer.fit([i for ent in pov_df["entities"].to_list() for i in ent])
    entity_feat = normalize(
        np.vstack(
            [
                np.array(vectorizer.transform(row["entities"]).mean(axis=0))
                for row in all_centroids
            ]
        )
    )

    ## make pairs
    pairs = np.array(
        [
            (row["cluster_id"], n)
            for row in all_centroids
            for n in row["neighbors"]
            if n != row["cluster_id"]
        ]
    )

    ## cosine similarity
    sims = np.einsum("ij,ij->i", entity_feat[pairs[:, 0]], entity_feat[pairs[:, 1]])
    merge_pairs = pairs[sims > 0.75]

    ## assign new cluster_id
    n = len(all_centroids)
    adj = csr_matrix(
        (np.ones(len(merge_pairs)), (merge_pairs[:, 0], merge_pairs[:, 1])), shape=(n, n)
    )
    _, new_labels = connected_components(adj, directed=False)
    cluster_id_map = centroid_df.select("cluster_id").with_columns(
        pl.Series(new_labels).alias("new_cluster_id")
    )

    pov_df = (
        pov_df.join(cluster_id_map, on="cluster_id", how="left")
        .with_columns(pl.col("new_cluster_id").alias("cluster_id"))
        .drop("new_cluster_id")
    )


def main(args):
    df = pl.read_csv(args.input)

    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    main(args)
