"""
GRES Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import math
import os

from functools import reduce
import operator

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import DatasetEvaluators, verify_results

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from gres_model import (
    PlantSegEvaluator,
    PlantSegMapper,
    RefCOCOMapper,
    ReferEvaluator,
    add_maskformer2_config,
    add_refcoco_config
)
from tools.server_assets import ensure_swin_weights


class BestMetricCheckpointer(hooks.HookBase):
    def __init__(self, checkpointer, metric_name, mode="max", file_prefix="model_best"):
        self._checkpointer = checkpointer
        self._metric_name = metric_name
        self._mode = mode
        self._file_prefix = file_prefix
        self._best_metric = None
        self._best_iter = None
        self._logger = logging.getLogger(__name__)

    def _is_better(self, metric_value):
        if self._best_metric is None:
            return True
        if self._mode == "min":
            return metric_value < self._best_metric
        return metric_value > self._best_metric

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if self.trainer.cfg.TEST.EVAL_PERIOD <= 0:
            return
        if not is_final and next_iter % self.trainer.cfg.TEST.EVAL_PERIOD != 0:
            return

        latest_metrics = self.trainer.storage.latest()
        if self._metric_name not in latest_metrics:
            self._logger.warning("Metric %s not found in storage after evaluation.", self._metric_name)
            return

        metric_value = latest_metrics[self._metric_name][0]
        if not self._is_better(metric_value):
            return

        self._best_metric = metric_value
        self._best_iter = next_iter
        self.trainer.storage.put_scalar("best_metric", metric_value, smoothing_hint=False)
        self._checkpointer.save(
            self._file_prefix,
            iteration=next_iter,
            best_metric_name=self._metric_name,
            best_metric_value=metric_value,
        )
        self._logger.info(
            "Saved %s at iteration %d with %s=%.6f",
            self._file_prefix,
            next_iter,
            self._metric_name,
            metric_value,
        )


class Trainer(DefaultTrainer):
    @classmethod
    def _build_mapper(cls, cfg, is_train):
        if cfg.INPUT.DATASET_MAPPER_NAME == "refcoco":
            return RefCOCOMapper(cfg, is_train)
        if cfg.INPUT.DATASET_MAPPER_NAME == "plantseg":
            return PlantSegMapper(cfg, is_train)
        raise NotImplementedError(f"Unsupported dataset mapper {cfg.INPUT.DATASET_MAPPER_NAME}")

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        metadata = MetadataCatalog.get(dataset_name)
        evaluator_type = metadata.get("evaluator_type")

        if evaluator_type == "refer":
            evaluator_list.append(
                ReferEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        elif evaluator_type == "plantseg":
            evaluator_list.append(
                PlantSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    save_masks=cfg.TEST.SAVE_PREDICTION_MASKS and metadata.get("split") == "test",
                    mask_subdir=cfg.TEST.PREDICTION_MASK_DIR,
                )
            )
        else:
            raise NotImplementedError(f"No evaluator for dataset {dataset_name} with type {evaluator_type}")

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = cls._build_mapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = cls._build_mapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        trainer_hooks = super().build_hooks()
        if not self.cfg.TEST.BEST_METRIC:
            return trainer_hooks

        for index, current_hook in enumerate(trainer_hooks):
            if isinstance(current_hook, hooks.EvalHook):
                trainer_hooks.insert(
                    index + 1,
                    BestMetricCheckpointer(
                        self.checkpointer,
                        self.cfg.TEST.BEST_METRIC,
                        mode=self.cfg.TEST.BEST_MODE,
                    ),
                )
                break
        return trainer_hooks

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if "text_encoder" in module_name:
                    continue
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0

                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm

                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        hyperparams = copy.copy(defaults)
        params.append({"params": reduce(operator.concat,
                                        [[p for p in model.text_encoder.encoder.layer[i].parameters()
                                          if p.requires_grad] for i in range(10)]), 
                        **hyperparams
                     })

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    _apply_epoch_schedule(cfg)
    _ensure_model_weights(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="referring")
    return cfg


def _apply_epoch_schedule(cfg):
    if cfg.SOLVER.MAX_EPOCHS <= 0:
        return

    train_dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / cfg.SOLVER.IMS_PER_BATCH))
    cfg.defrost()
    cfg.SOLVER.STEPS_PER_EPOCH = steps_per_epoch
    cfg.SOLVER.MAX_ITER = steps_per_epoch * cfg.SOLVER.MAX_EPOCHS
    if cfg.SOLVER.EPOCH_MILESTONES:
        cfg.SOLVER.STEPS = tuple(int(steps_per_epoch * epoch) for epoch in cfg.SOLVER.EPOCH_MILESTONES)
    if cfg.TEST.EVAL_PERIOD_EPOCHS > 0:
        cfg.TEST.EVAL_PERIOD = int(steps_per_epoch * cfg.TEST.EVAL_PERIOD_EPOCHS)
    if cfg.SOLVER.CHECKPOINT_PERIOD_EPOCHS > 0:
        cfg.SOLVER.CHECKPOINT_PERIOD = int(steps_per_epoch * cfg.SOLVER.CHECKPOINT_PERIOD_EPOCHS)


def _ensure_model_weights(cfg):
    resolved_weights = ensure_swin_weights(cfg.MODEL.WEIGHTS)
    if resolved_weights == cfg.MODEL.WEIGHTS:
        return
    cfg.defrost()
    cfg.MODEL.WEIGHTS = resolved_weights


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
