import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.server_assets import build_msdeformattn, detect_plantseg_root, ensure_runtime_assets


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare server-side model assets for plantseg training.")
    parser.add_argument(
        "--model-weights",
        default="models/plantseg/swin_base_patch4_window12_384_22k.pkl",
        help="Relative or absolute path to the converted Detectron2 checkpoint.",
    )
    parser.add_argument(
        "--bert-type",
        default="models/plantseg/bert-base-uncased",
        help="Hugging Face model id used by the tokenizer and text encoder.",
    )
    parser.add_argument(
        "--build-op",
        action="store_true",
        help="Build and install the MultiScaleDeformableAttention extension.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the resolved data and asset locations without downloading or building anything.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_plan:
        print(f"plantseg_root={detect_plantseg_root()}")
        print(f"model_weights={args.model_weights}")
        print(f"bert_type={args.bert_type}")
        print(f"build_op={args.build_op}")
        return

    resolved_weights, resolved_bert = ensure_runtime_assets(args.model_weights, args.bert_type, build_op=False)
    print(f"prepared_model_weights={resolved_weights}")
    print(f"prepared_bert_type={resolved_bert}")
    if args.build_op:
        build_msdeformattn()
        print("built_msdeformattn=true")


if __name__ == "__main__":
    main()
