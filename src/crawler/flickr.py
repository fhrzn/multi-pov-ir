import os
import threading
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Optional

import flickr_api
import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
flickr_api.set_keys(
    api_key=os.getenv("FLICKR_API_KEY"), api_secret=os.getenv("FLICKR_API_SECRET")
)

# flickr.photos.search errors with "Too many tags in query" past this; keep
# a safety margin under it since the exact undocumented cutoff can vary.
MAX_TAGS_PER_QUERY = 20

# Flickr's documented cap is ~3600 calls/hour per key (~1 req/s) for
# api.flickr.com REST calls. Only Photo.search() hits that endpoint here --
# with these extras every field FlickrSpot needs comes back inline per
# *page*, so there's no more per-photo getInfo() call to also throttle.
API_MAX_QPS = 1.0
SEARCH_EXTRAS = "date_taken,tags,geo,url_n"

# Image bytes come from live.staticflickr.com, a plain CDN unrelated to the
# api.flickr.com quota above, so downloads are safe to run concurrently.
IMAGE_DOWNLOAD_WORKERS = 8
IMAGE_SIZE_LABEL = "Small 320"  # matches the url_n column

# Safety net: some poi_name_tags entries embed a literal comma from the
# place name itself (e.g. "nagara bridge, osaka"), which _chunk_tags can't
# tell apart from a real tag separator. That fragment can turn into a single
# overly generic tag (e.g. "osaka") matching well over a million photos, so
# without a cap a single entity can take hours to walk at 1 req/s.
MAX_PHOTOS_PER_CHUNK = 500


class RateLimiter:
    """Thread-safe gate that spaces calls >= 1/max_qps apart."""

    def __init__(self, max_qps: float):
        self._min_interval = 1.0 / max_qps
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            sleep_for = self._min_interval - (time.monotonic() - self._last_call)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


_api_limiter = RateLimiter(API_MAX_QPS)


def _throttled_search(*args, **kwargs):
    _api_limiter.wait()
    return flickr_api.Photo.search(*args, **kwargs)


def _chunk_tags(keywords: str, size: int = MAX_TAGS_PER_QUERY) -> list[str]:
    """Split a comma-delimited tag string into <= `size`-tag chunks."""
    tags = [t.strip() for t in keywords.split(",") if t.strip()]
    return [",".join(tags[i : i + size]) for i in range(0, len(tags), size)]


@dataclass(kw_only=True)
class FlickrSpot:
    category: str = None
    id: str
    owner: str
    datetaken: Optional[int] = None
    title: str = ""
    tags: str = ""
    latitude: Optional[str] = None
    longitude: Optional[str] = None
    place_id: Optional[str] = None
    url_n: Optional[str] = None
    img_id: Optional[str] = None
    text: Optional[str] = None

    def __post_init__(self):
        self.img_id = (
            "-".join(
                [
                    self.category,
                    str(self.latitude),
                    str(self.longitude),
                    str(self.datetaken),
                    self.owner,
                    self.id,
                    "n",
                ]
            )
            + ".jpg"
        )
        self.text = f"{self.title} {self.tags}".strip()

    @classmethod
    def from_photo(cls, photo: flickr_api.Photo, category: str) -> Optional["FlickrSpot"]:
        # extras="geo" always sets latitude/longitude/accuracy; accuracy "0"
        # means the photo simply has no geotag (same photos getInfo()'s
        # `location` key used to omit entirely).
        if str(getattr(photo, "accuracy", "0")) == "0":
            return None

        datetaken = getattr(photo, "datetaken", None)
        if isinstance(datetaken, str):
            datetaken = int(
                datetime.strptime(datetaken, "%Y-%m-%d %H:%M:%S").timestamp()
            )

        tags = getattr(photo, "tags", None)
        if isinstance(tags, list):
            # some responses return Tag objects, others a flat string
            tags = " ".join(getattr(t, "text", t) for t in tags)

        return cls(
            category=category,
            id=photo.id,
            owner=photo.owner.id,
            title=getattr(photo, "title", ""),
            datetaken=datetaken,
            tags=tags or "",
            latitude=getattr(photo, "latitude", None),
            longitude=getattr(photo, "longitude", None),
            place_id=getattr(photo, "place_id", None),
            url_n=getattr(photo, "url_n", None),
        )

    def to_row(self) -> dict:
        """Dict in dataset/csv/spots/*.csv column order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Explicit dtypes (matching src/preprocess/splatone_to_csv.py's schema).
# Without this, pl.DataFrame(list[dict]) can mis-infer a column as Null when
# the first rows all have e.g. place_id=None, then crash once a real string
# value shows up later in the batch.
FLICKR_SPOT_SCHEMA = {
    "category": pl.Utf8,
    "id": pl.Utf8,
    "owner": pl.Utf8,
    "datetaken": pl.Int64,
    "title": pl.Utf8,
    "tags": pl.Utf8,
    "latitude": pl.Utf8,
    "longitude": pl.Utf8,
    "place_id": pl.Utf8,
    "url_n": pl.Utf8,
    "img_id": pl.Utf8,
    "text": pl.Utf8,
}


def _download_image(photo: flickr_api.Photo, path: str) -> None:
    try:
        photo.save(path, size_label=IMAGE_SIZE_LABEL)
    except Exception as e:
        print(f"skip image {path}: {e}")


def paginate(category: str, keywords: str, entity_id: int):
    csv_dir = f"./dataset/csv/additional_spots/{category}/"
    img_dir = f"./dataset/images/test_{category}/"
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    photos = []
    seen_ids = set()
    download_jobs = []

    for tag_chunk in _chunk_tags(keywords):
        pagination = flickr_api.Walker(
            _throttled_search, tags=tag_chunk, extras=SEARCH_EXTRAS
        )
        total_matches = len(pagination)
        chunk_cap = min(total_matches, MAX_PHOTOS_PER_CHUNK)
        if total_matches > MAX_PHOTOS_PER_CHUNK:
            print(
                f"entity {entity_id} ({category}): '{tag_chunk}' matched "
                f"{total_matches} photos -- capping at {MAX_PHOTOS_PER_CHUNK}"
            )

        search_bar = tqdm(
            total=chunk_cap,
            desc=f"entity {entity_id} search",
            leave=False,
        )
        for i, p in enumerate(pagination):
            if i >= MAX_PHOTOS_PER_CHUNK:
                break
            search_bar.update(1)
            if p.id in seen_ids:
                continue
            seen_ids.add(p.id)

            item = FlickrSpot.from_photo(p, category)
            if not item:
                continue

            photos.append(item.to_row())
            download_jobs.append((p, os.path.join(img_dir, item.img_id)))
        search_bar.close()

    if download_jobs:
        with ThreadPoolExecutor(max_workers=IMAGE_DOWNLOAD_WORKERS) as pool:
            list(
                tqdm(
                    pool.map(lambda job: _download_image(*job), download_jobs),
                    total=len(download_jobs),
                    desc=f"entity {entity_id} download",
                    leave=False,
                )
            )

    if photos:
        pl.DataFrame(photos, schema=FLICKR_SPOT_SCHEMA).write_csv(
            os.path.join(csv_dir, f"{entity_id}.csv")
        )


def _merge_category_csvs(csv_root: str = "./dataset/csv/additional_spots") -> None:
    """Combine each category's per-entity CSVs into one <category>.csv.

    Per-entity files are kept as-is (cheap crash resilience across a long
    run over hundreds of entities); this just concatenates them afterward.
    """
    if not os.path.isdir(csv_root):
        return

    for category in sorted(os.listdir(csv_root)):
        category_dir = os.path.join(csv_root, category)
        if not os.path.isdir(category_dir):
            continue

        parts = sorted(f for f in os.listdir(category_dir) if f.endswith(".csv"))
        if not parts:
            continue

        merged = pl.concat(
            [
                pl.read_csv(
                    os.path.join(category_dir, f), schema_overrides=FLICKR_SPOT_SCHEMA
                )
                for f in parts
            ]
        )
        merged.write_csv(os.path.join(csv_root, f"{category}.csv"))


def main(args):
    df = pl.read_csv(args.from_file)
    rows = df.select(["category", "poi_name_tags", "entity_id"]).to_dicts()

    for row in tqdm(rows):
        try:
            paginate(row["category"], row["poi_name_tags"], row["entity_id"])
        except Exception as e:
            print(f"skip entity {row['entity_id']} ({row['category']}): {e}")
            continue

    _merge_category_csvs()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--from-file", type=str)

    args = parser.parse_args()
    main(args)
