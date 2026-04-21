import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .plantseg import load_plantseg_json


def _default_plantseg_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "plantseg")
    )


def register_plantseg(root=None, caption_index=2, ignore_false_healthy=True):
    root = os.path.abspath(root or os.getenv("PLANTSEG_ROOT", _default_plantseg_root()))
    json_file = os.path.join(root, "main.json")

    for split in ("train", "val", "test"):
        dataset_name = f"plantseg_{split}"
        if dataset_name in DatasetCatalog.list():
            continue
        DatasetCatalog.register(
            dataset_name,
            lambda root=root, split=split, caption_index=caption_index, ignore_false_healthy=ignore_false_healthy: load_plantseg_json(
                root,
                split,
                caption_index=caption_index,
                ignore_false_healthy=ignore_false_healthy,
            ),
        )
        MetadataCatalog.get(dataset_name).set(
            evaluator_type="plantseg",
            dataset_name="plantseg",
            split=split,
            root=root,
            json_file=json_file,
        )


register_plantseg()
