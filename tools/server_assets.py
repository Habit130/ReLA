import os
import pickle
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models" / "plantseg"
DEFAULT_BERT_DIR = MODELS_ROOT / "bert-base-uncased"

SWIN_SPECS = {
    "swin_base_patch4_window12_384_22k.pkl": {
        "raw_name": "swin_base_patch4_window12_384_22k.pth",
        "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
    },
    "swin_tiny_patch4_window7_224.pkl": {
        "raw_name": "swin_tiny_patch4_window7_224.pth",
        "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    },
}


def resolve_repo_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_bert_source(model_name):
    preferred_local = DEFAULT_BERT_DIR
    if preferred_local.exists():
        return str(preferred_local)

    candidate = resolve_repo_path(model_name)
    if candidate.exists():
        return str(candidate)

    return model_name


def download_file(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    with urllib.request.urlopen(url) as response, open(destination, "wb") as handle:
        handle.write(response.read())
    return destination


def convert_swin_checkpoint(input_path, output_path):
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = torch.load(str(input_path), map_location="cpu")["model"]
    payload = {"model": model, "__author__": "third_party", "matching_heuristics": True}
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)
    return output_path


def ensure_swin_weights(model_weights):
    if not model_weights or model_weights.startswith("detectron2://"):
        return model_weights

    target_path = resolve_repo_path(model_weights)
    if target_path.exists():
        return str(target_path)

    spec = SWIN_SPECS.get(target_path.name)
    if spec is None:
        return str(target_path)

    raw_path = target_path.parent / spec["raw_name"]
    download_file(spec["url"], raw_path)
    convert_swin_checkpoint(raw_path, target_path)
    return str(target_path)


def ensure_bert_assets(model_name):
    from transformers import BertModel, BertTokenizer

    resolved_model_name = resolve_bert_source(model_name)
    BertTokenizer.from_pretrained(resolved_model_name)
    BertModel.from_pretrained(resolved_model_name)
    return resolved_model_name


def build_msdeformattn():
    ops_dir = REPO_ROOT / "gres_model" / "modeling" / "pixel_decoder" / "ops"
    subprocess.check_call([sys.executable, "setup.py", "build", "install"], cwd=str(ops_dir))


def ensure_runtime_assets(model_weights, bert_type, build_op=False):
    resolved_weights = ensure_swin_weights(model_weights)
    resolved_bert = ensure_bert_assets(bert_type)
    if build_op:
        build_msdeformattn()
    return resolved_weights, resolved_bert


def default_plantseg_root():
    return (REPO_ROOT.parent / "plantseg").resolve()


def detect_plantseg_root():
    return Path(os.getenv("PLANTSEG_ROOT", str(default_plantseg_root()))).resolve()
