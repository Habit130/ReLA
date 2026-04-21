import json
import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager


def _safe_divide(numerator, denominator):
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


class PlantSegEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed=True, output_dir=None, save_masks=False, mask_subdir="test_masks"):
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._save_masks = save_masks
        self._mask_subdir = mask_subdir
        self._cpu_device = torch.device("cpu")

    def reset(self):
        self._records = []

    def process(self, inputs, outputs):
        for input_item, output_item in zip(inputs, outputs):
            pred_mask = output_item["ref_seg"].argmax(dim=0).to(self._cpu_device).numpy().astype(np.uint8)
            gt_mask = input_item["gt_mask_merged"].to(self._cpu_device).numpy().astype(np.uint8)
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[0]

            pred_mask = (pred_mask > 0).astype(np.uint8)
            gt_mask = (gt_mask > 0).astype(np.uint8)

            tp = int(np.logical_and(pred_mask == 1, gt_mask == 1).sum())
            fp = int(np.logical_and(pred_mask == 1, gt_mask == 0).sum())
            fn = int(np.logical_and(pred_mask == 0, gt_mask == 1).sum())
            tn = int(np.logical_and(pred_mask == 0, gt_mask == 0).sum())

            self._records.append(
                {
                    "image_id": input_item["image_id"],
                    "mask_relpath": input_item["mask_relpath"],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
            )

            if self._save_masks and self._output_dir:
                save_path = os.path.join(self._output_dir, self._mask_subdir, input_item["mask_relpath"])
                PathManager.mkdirs(os.path.dirname(save_path))
                Image.fromarray(pred_mask * 255).save(save_path)

    def evaluate(self):
        if self._distributed:
            synchronize()
            records = all_gather(self._records)
            records = [item for chunk in records for item in chunk]
            if not is_main_process():
                return
        else:
            records = self._records

        tp = sum(item["tp"] for item in records)
        fp = sum(item["fp"] for item in records)
        fn = sum(item["fn"] for item in records)
        tn = sum(item["tn"] for item in records)

        iou_fg = _safe_divide(tp, tp + fp + fn)
        dice_fg = _safe_divide(2 * tp, 2 * tp + fp + fn)
        recall_fg = _safe_divide(tp, tp + fn)
        iou_bg = _safe_divide(tn, tn + fp + fn)
        acc_bg = _safe_divide(tn, tn + fp)
        m_iou = (iou_fg + iou_bg) / 2.0
        m_acc = (recall_fg + acc_bg) / 2.0

        metrics = {
            "IoU": iou_fg * 100.0,
            "Dice": dice_fg * 100.0,
            "Recall": recall_fg * 100.0,
            "mIoU": m_iou * 100.0,
            "mACC": m_acc * 100.0,
        }

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            summary_path = os.path.join(self._output_dir, f"{self._dataset_name}_summary.json")
            payload = {
                "dataset_name": self._dataset_name,
                "num_samples": len(records),
                "metrics": metrics,
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            }
            with PathManager.open(summary_path, "w") as handle:
                handle.write(json.dumps(payload, indent=2))

        return OrderedDict([("plantseg", metrics)])
