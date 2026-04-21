import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def parse_args():
    parser = argparse.ArgumentParser(description="Validate the Linux server environment for plantseg training.")
    parser.add_argument("--config-file", default=None, help="Optional config file to load and inspect.")
    parser.add_argument("opts", nargs="*", help="Additional Detectron2 config overrides.")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    import detectron2
    import transformers

    print(f"torch={torch.__version__}")
    print(f"detectron2={detectron2.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_version={torch.version.cuda}")

    import MultiScaleDeformableAttention  # noqa: F401

    print("msdeformattn_import=true")

    if args.config_file:
        from detectron2.config import get_cfg
        from detectron2.data import DatasetCatalog

        from gres_model import add_maskformer2_config, add_refcoco_config

        cfg = get_cfg()
        add_maskformer2_config(cfg)
        add_refcoco_config(cfg)
        cfg.merge_from_file(args.config_file)
        if args.opts:
            cfg.merge_from_list(args.opts)

        print(f"config_file={Path(args.config_file).resolve()}")
        for dataset_name in cfg.DATASETS.TRAIN:
            dataset = DatasetCatalog.get(dataset_name)
            print(f"train_dataset={dataset_name}:{len(dataset)}")
        for dataset_name in cfg.DATASETS.TEST:
            dataset = DatasetCatalog.get(dataset_name)
            print(f"test_dataset={dataset_name}:{len(dataset)}")


if __name__ == "__main__":
    main()
