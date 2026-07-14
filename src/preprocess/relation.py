import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocess.utils import (
    compute_entity_feat,
    normalize_text,
    spatial_radius_query,
)


def main(args):
    entity_df = (
        pl.read_csv(args.entity)
        .unique(subset=["wikimedia_url"])
        .with_columns(
            pl.col("poi_name_tags")
            .map_elements(
                lambda x: [n for names in json.loads(x).values() for n in names],
                return_dtype=pl.List(str),
            )
            .alias("poi_name_tags")
        )
        .with_columns(
            pl.col("poi_name_tags").list.eval(
                pl.element()
                .str.replace(r"^[^:]+:", "")
                .map_elements(normalize_text, return_dtype=pl.String)
            )
        )
    )
    cluster_df = pl.read_csv(args.input).with_columns(
        [
            pl.col("member").str.split(",").list.eval(pl.element().cast(pl.Int64)),
            pl.col("entities")
            .str.split(",")
            .list.eval(pl.element().filter(pl.element() != "")),
        ]
    )

    # identify poi-pov neighbors
    ## spatial proximity
    entity_coords = entity_df[["latitude", "longitude"]].to_numpy()
    cluster_coords = cluster_df[["latitude", "longitude"]].to_numpy()
    indices, distances = spatial_radius_query(
        entity_coords, cluster_coords, radius_km=0.1
    )

    entity_df = entity_df.with_columns(
        pl.Series(
            "nearby_pov_cluster", [i.tolist() for i in indices], dtype=pl.List(pl.Int64)
        )
    )

    ## lexical confirmation
    ### vectorize
    vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    corpus = [i for ent in cluster_df["entities"].to_list() if ent for i in ent] + [
        i for ent in entity_df["poi_name_tags"].to_list() if ent for i in ent
    ]
    vectorizer.fit(corpus)
    entity_feat = compute_entity_feat(
        [ent for ent in entity_df["poi_name_tags"].to_list()], vectorizer
    )
    cluster_feat = compute_entity_feat(
        [ent for ent in cluster_df["entities"].to_list()], vectorizer
    )

    ### make pairs
    pairs = np.array(
        [
            (row["entity_id"], n)
            for row in entity_df.to_dicts()
            for n in row["nearby_pov_cluster"]
        ]
    )

    ### cosine similarity
    sims = np.einsum("ij,ij->i", entity_feat[pairs[:, 0]], cluster_feat[pairs[:, 1]])
    valid_pairs = pairs[sims > 0.5]
    print(pairs)
    print(sims)
    entity_to_cluster = defaultdict(list)
    for eid, cid in valid_pairs:
        entity_to_cluster[int(eid)].append(int(cid))

    # re-assign valid nearby poi-pov neighbors
    entity_df = entity_df.with_columns(
        pl.Series(
            "nearby_pov_cluster",
            [
                entity_to_cluster.get(eid, [])
                for eid in entity_df["entity_id"].to_list()
            ],
            dtype=pl.List(pl.Int64),
        )
    )

    entity_df.with_columns(
        [
            pl.col("nearby_pov_cluster").cast(pl.List(pl.String)).list.join(","),
            pl.col("poi_name_tags").list.join(","),
        ]
    ).write_csv(args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    main(args)
