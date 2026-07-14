from argparse import ArgumentParser
from typing import List

import numpy as np
import polars as pl
from gliner import GLiNER
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess.utils import (
    compute_entity_feat,
    extract_entities,
    spatial_radius_query,
    spherical_centroid,
)
from src.utils import get_device

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


def get_centroid(df: pl.DataFrame):
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
    indices, distances = spatial_radius_query(
        centroid_coords, centroid_coords, radius_km=1.0
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
    extracted_ner = extract_entities(
        pov_df["text"].to_list(), ner_labels, model, threshold
    )
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
    entity_feat = compute_entity_feat(
        [row["entities"] for row in all_centroids], vectorizer
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
        (np.ones(len(merge_pairs)), (merge_pairs[:, 0], merge_pairs[:, 1])),
        shape=(n, n),
    )
    _, new_labels = connected_components(adj, directed=False)
    cluster_id_map = centroid_df.select("cluster_id").with_columns(
        pl.Series(new_labels).alias("new_cluster_id")
    )

    # re-assign the new cluster id
    pov_df = (
        pov_df.join(cluster_id_map, on="cluster_id", how="left")
        .with_columns(pl.col("new_cluster_id").alias("cluster_id"))
        .drop("new_cluster_id")
    )

    centroid_df = (
        centroid_df.join(cluster_id_map, on="cluster_id", how="left")
        .group_by("new_cluster_id")
        .agg(
            [
                pl.col("member").explode(),
                pl.col("latitude"),
                pl.col("longitude"),
                pl.col("entities").explode(),
            ]
        )
        .with_columns(
            pl.struct(["latitude", "longitude"])
            .map_elements(
                lambda row: spherical_centroid(
                    np.array(row["latitude"]), np.array(row["longitude"])
                ),
                return_dtype=pl.List(pl.Float64),
            )
            .alias("centroid")
        )
        .with_columns(
            [
                pl.col("centroid").list.get(0).alias("latitude"),
                pl.col("centroid").list.get(1).alias("longitude"),
                pl.col("new_cluster_id").alias("cluster_id"),
            ]
        )
        .drop(["centroid", "new_cluster_id"])
        .select(["cluster_id", "latitude", "longitude", "member", "entities"])
        .sort("cluster_id")
    )

    return pov_df, centroid_df


def main(args):
    df = pl.read_csv(args.input).filter(pl.col("country_code") == "JP")
    pov_df = assign_cluster(df)
    centroid_df = get_centroid(pov_df)
    pov_df, centroid_df = merge_cluster(pov_df, centroid_df, args.ner_labels)

    pov_df.with_columns(pl.col("entities").list.join(",")).write_csv(
        "dataset/csv/bridge_spots.csv"
    )
    centroid_df.with_columns(
        [
            pl.col("member").cast(pl.List(pl.String)).list.join(","),
            pl.col("entities").list.join(","),
        ]
    ).write_csv("dataset/csv/bridge_clusters.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--ner-labels", required=True, nargs="+")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    main(args)
