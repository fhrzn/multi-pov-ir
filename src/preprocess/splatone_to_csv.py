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

    return pl.DataFrame(df)


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
