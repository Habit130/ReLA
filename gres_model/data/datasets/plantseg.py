import json
import os

from detectron2.utils.file_io import PathManager


def load_plantseg_json(dataset_root, split, caption_index=3, ignore_false_healthy=True):
    json_file = os.path.join(dataset_root, "main.json")
    json_file = PathManager.get_local_path(json_file)

    with PathManager.open(json_file, "r") as handle:
        annotations = json.load(handle)

    dataset_dicts = []
    for item in annotations:
        if item.get("split") != split:
            continue
        if ignore_false_healthy and item.get("false_healthy_ann"):
            continue

        captions = item.get("caption", [])
        if len(captions) <= caption_index:
            raise ValueError(f"Sample {item.get('id')} does not have caption[{caption_index}]")

        record = {
            "source": "plantseg",
            "image_id": item["id"],
            "file_name": os.path.join(dataset_root, item["image"]),
            "mask_file_name": os.path.join(dataset_root, item["mask"]),
            "mask_relpath": item["mask"],
            "sentence": {
                "raw": captions[caption_index],
                "caption_index": caption_index,
            },
            "disease_label": item.get("disease_label"),
            "empty": False,
        }
        dataset_dicts.append(record)

    return dataset_dicts
