import copy
import logging

import numpy as np
import torch
from PIL import Image
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from transformers import BertTokenizer


__all__ = ["PlantSegMapper"]


def build_transform_train(cfg):
    return [T.Resize((cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE))]


def build_transform_test(cfg):
    return [T.Resize((cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE))]


class PlantSegMapper:
    @configurable
    def __init__(self, is_train=True, *, tfm_gens, image_format, bert_type, max_tokens):
        self.is_train = is_train
        self.tfm_gens = tfm_gens
        self.img_format = image_format
        self.bert_type = bert_type
        self.max_tokens = max_tokens
        logging.getLogger(__name__).info("Full TransformGens used: %s", self.tfm_gens)
        logging.getLogger(__name__).info("Loading BERT tokenizer: %s...", self.bert_type)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_type)

    @classmethod
    def from_config(cls, cfg, is_train=True):
        return {
            "is_train": is_train,
            "tfm_gens": build_transform_train(cfg) if is_train else build_transform_test(cfg),
            "image_format": cfg.INPUT.FORMAT,
            "bert_type": cfg.REFERRING.BERT_TYPE,
            "max_tokens": cfg.REFERRING.MAX_TOKENS,
        }

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        if "height" in dataset_dict and "width" in dataset_dict:
            utils.check_image_size(dataset_dict, image)
        else:
            dataset_dict["height"], dataset_dict["width"] = image.shape[:2]

        mask = np.array(Image.open(dataset_dict["mask_file_name"]).convert("L"), dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)

        padding_mask = np.ones(image.shape[:2], dtype=np.uint8)
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        mask = transforms.apply_segmentation(mask)
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]
        mask = (mask > 0).astype(np.uint8)
        gt_masks = torch.as_tensor(np.ascontiguousarray(mask[None, ...]), dtype=torch.uint8)

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        instances = Instances(image_shape)
        instances.gt_masks = gt_masks
        instances.gt_classes = torch.zeros((gt_masks.shape[0],), dtype=torch.int64)
        instances.gt_boxes = BitMasks(gt_masks).get_bounding_boxes()

        if self.is_train:
            dataset_dict["instances"] = instances
        else:
            dataset_dict["gt_mask"] = gt_masks

        dataset_dict["empty"] = False
        dataset_dict["gt_mask_merged"] = gt_masks

        sentence_raw = dataset_dict["sentence"]["raw"]
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens
        input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
        input_ids = input_ids[:self.max_tokens]
        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)] = [1] * len(input_ids)
        dataset_dict["lang_tokens"] = torch.tensor(padded_input_ids).unsqueeze(0)
        dataset_dict["lang_mask"] = torch.tensor(attention_mask).unsqueeze(0)

        return dataset_dict
