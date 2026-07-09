import json

import gradio as gr
import polars as pl
import os

BASEPATH = "/home/affahrizain/projects/datasets/geotir"
OUTPUT_PATH = os.path.join(BASEPATH, "jp_gldv_mp16_scenetags_human.jsonl")


def load_csv(filepath: str):
    df = pl.read_csv(filepath)
    stats = load_stats(df)
    state = {"data": df}

    if not os.path.exists(OUTPUT_PATH):
        return state, stats

    state["result_data"] = pl.read_ndjson(OUTPUT_PATH)
    return state, stats


def load_stats(df: pl.DataFrame):
    scene_stats = df.group_by("scene_type").agg(pl.len())
    scene_stats = dict(zip(scene_stats["scene_type"], scene_stats["len"]))
    viewpoint_stats = df.group_by("viewpoint_type").agg(pl.len())
    viewpoint_stats = dict(
        zip(viewpoint_stats["viewpoint_type"], viewpoint_stats["len"])
    )

    str_out = (
        "=== SCENE STATS ===\n"
        + str(scene_stats)
        + "\n\n"
        + "=== VIEWPOINT STATS ===\n"
        + str(viewpoint_stats)
    )

    return str_out


def load_images(state: dict):
    df = state["data"]
    if state.get("result_data") is None:
        item = df.sample(1).row(0, named=True)
    else:
        unique_ids = state["result_data"]["id"].unique().to_list()
        item = df.filter(~pl.col("id").is_in(unique_ids)).sample(1).row(0, named=True)

    state["annot_item"] = item
    return (
        state,
        os.path.join(BASEPATH, item["dataset"], item["src"], f"{item['id']}.jpg"),
    )


def write_annotation(state: dict, scene_type: str, viewpoint_type: str):
    row = state["annot_item"]
    row["human_scene_type"] = scene_type
    row["human_viewpoint_type"] = viewpoint_type

    with open(OUTPUT_PATH, "a") as f:
        f.write(json.dumps(row))
        f.write("\n")

    if state.get("result_data") is None:
        state["result_data"] = pl.read_ndjson(OUTPUT_PATH)
    else:
        state["result_data"] = pl.concat(
            [state["result_data"], pl.DataFrame([row])], how="vertical_relaxed"
        )

    attr = load_images(state)
    return attr


### component UI
with gr.Blocks() as demo:
    state = gr.State()
    gr.Markdown("# Viewpoint annotator tool")
    gr.Row()

    with gr.Column():
        with gr.Row(equal_height=True):
            file_upload = gr.File(file_count="single", file_types=[".csv"])
            with gr.Column():
                file_stats = gr.TextArea(lines=5, label="File Statistics")
                btn_upload = gr.Button("Load CSV", variant="primary")

        gr.Row(height=20)
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Image")
                img_viewer = gr.Image(type="filepath", interactive=False)
            with gr.Column():
                gr.Markdown("## Annotation")
                scene_choice = gr.Radio(
                    choices=["indoor", "outdoor"], label="Scene Type"
                )
                viewpoint_choice = gr.Radio(
                    choices=["ground", "aerial"], label="Viewpoint Type"
                )

        with gr.Row():
            btn_skip_img = gr.Button("Skip")
            btn_submit_annot = gr.Button("Submit", variant="primary")

    ### component actions
    file_upload.upload(fn=load_csv, inputs=file_upload, outputs=[state, file_stats])
    btn_upload.click(fn=load_images, inputs=state, outputs=[state, img_viewer])
    btn_skip_img.click(fn=load_images, inputs=state, outputs=[state, img_viewer])
    btn_submit_annot.click(
        fn=write_annotation,
        inputs=[state, scene_choice, viewpoint_choice],
        outputs=[state, img_viewer],
    )


if __name__ == "__main__":
    demo.launch(
        allowed_paths=[
            BASEPATH,
            os.path.join(BASEPATH, "mp16", "images"),
            os.path.join(BASEPATH, "gldv2-full", "train"),
            os.path.join(BASEPATH, "gldv2-full", "index"),
        ]
    )
