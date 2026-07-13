import argparse
import math
from typing import Iterable

import country_converter as coco
import polars as pl
import pycountry
import pycountry_convert as pc
import reverse_geocoder as rg
from tqdm import tqdm


def _build_country_df() -> pl.DataFrame:
    """Build a country_code → country/region/subregion mapping.

    - country:    full name from pycountry
    - region:     continent from pycountry-convert
    - subregion:  UN region from country-converter (e.g. "Western Europe")
    """
    _cc = coco.CountryConverter()
    records = []
    for c in pycountry.countries:
        # Region (continent)
        try:
            continent_code = pc.country_alpha2_to_continent_code(c.alpha_2)
            region = pc.convert_continent_code_to_continent_name(continent_code)
        except Exception:
            region = None

        # Subregion (UN M.49 sub-region)
        try:
            subregion = _cc.convert(c.alpha_2, to="UNregion")
            subregion = None if subregion == "not found" else subregion
        except Exception:
            subregion = None

        records.append(
            {
                "country_code": c.alpha_2,
                "country": c.name,
                "region": region,
                "subregion": subregion,
            }
        )
    return pl.DataFrame(records)


# Country lookup table built at import time
df_cc = _build_country_df()


def _chunk_rows(
    rows: list[tuple[float, float]], chunk_size: int
) -> Iterable[list[tuple[float, float]]]:
    for i in range(0, len(rows), chunk_size):
        yield rows[i : i + chunk_size]


def reverse_geocode_df(
    df: pl.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    chunk_size: int = 250_000,
    rg_mode: int = 2,
) -> pl.DataFrame:
    # Keep null rows; only reverse geocode valid coordinates.
    valid = df.filter(
        pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null()
    ).with_columns(
        pl.col(lat_col).cast(pl.Float64),
        pl.col(lon_col).cast(pl.Float64),
    )

    if valid.is_empty():
        return df.with_columns(
            pl.lit(None).cast(pl.String).alias("country_code"),
            pl.lit(None).cast(pl.String).alias("country"),
            pl.lit(None).cast(pl.String).alias("region"),
            pl.lit(None).cast(pl.String).alias("subregion"),
        )

    # Huge speed-up for datasets with repeated coordinates.
    unique_coords = (
        valid.select([lat_col, lon_col])
        .unique(maintain_order=True)
        .with_row_index("coord_id")
    )

    coord_rows = unique_coords.select([lat_col, lon_col]).iter_rows()
    coord_rows = list(coord_rows)

    results: list[dict] = []
    total_chunks = math.ceil(len(coord_rows) / chunk_size)

    for batch in tqdm(
        _chunk_rows(coord_rows, chunk_size),
        total=total_chunks,
        desc="Reverse geocoding batches",
        unit="batch",
    ):
        # mode=2 uses multiprocessing in reverse_geocoder.
        batch_res = rg.search(batch, mode=rg_mode, verbose=False)
        results.extend(batch_res)

    rg_df = (
        pl.DataFrame(results)
        .with_row_index("coord_id")
        .rename(
            {
                "name": "city",
                "admin1": "rg_admin1",
                "admin2": "rg_admin2",
                "cc": "country_code",
                "lat": "rg_lat",
                "lon": "rg_lon",
            }
        )
        .with_columns(
            pl.col("rg_lat").cast(pl.Float64),
            pl.col("rg_lon").cast(pl.Float64),
        )
        .join(df_cc, on="country_code")
    )

    lookup = (
        unique_coords.join(rg_df, on="coord_id", how="left")
        .drop("coord_id")
        .select(
            [lat_col, lon_col, "country_code", "country", "region", "subregion", "city"]
        )
    )

    return df.with_columns(
        pl.col(lat_col).cast(pl.Float64),
        pl.col(lon_col).cast(pl.Float64),
    ).join(
        lookup,
        on=[lat_col, lon_col],
        how="left",
    )


def main(args):
    df = pl.read_csv(args.input)
    df_geocoded = reverse_geocode_df(df)
    df_geocoded.write_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    
    main(args)
