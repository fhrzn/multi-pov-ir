import json
from argparse import ArgumentParser
from datetime import datetime

import polars as pl
from tqdm.auto import tqdm

def to_csv(data: dict):
    col_list = [
        "id",
        "owner",
        "datetaken",
        "title",
        "tags",
        "latitude",
        "longitude",
        "place_id",
        "url_n",
    ]

    df = []
    for k, v in tqdm(data["geoJson"]["bulky"].items(), desc="collecting data.."):
        for item in v["features"]:
            body = {"category": k}
            for ik, iv in item["properties"].items():
                if ik in col_list:
                    if ik == "datetaken":
                        body[ik] = int(
                            datetime.strptime(iv, "%Y-%m-%d %H:%M:%S").timestamp()
                        )
                    else:
                        body[ik] = iv

            body["img_id"] = (
                "-".join(
                    [
                        k,
                        str(body.get("latitude")),
                        str(body.get("longitude")),
                        str(body.get("datetaken")),
                        body.get("owner"),
                        body.get("id"),
                        "n",
                    ]
                )
                + ".jpg"
            )
            body["text"] = " ".join([body.get("title"), body.get("tags")])

            df.append(body)

    schema = {
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
    return pl.DataFrame(df, schema=schema)


def main(args):
    data = json.loads(open(args.input, "r").read())
    df = to_csv(data)
    df.write_csv(args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    main(args)
