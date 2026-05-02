import yaml
import argparse
import torch

from src.inference.generate import load_model
from src.evaluation.metrics import perplexity
from src.data.tokenizer import load_tokenizer
from src.utils.logging import get_logger

logger = get_logger(__name__, "logs/evaluate.log")


def main(config_path: str, checkpoint: str, tokenizer_name: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint, cfg["model"], device)
    tokenizer = load_tokenizer(tokenizer_name)

    logger.info(f"Loaded model from {checkpoint}")
    # Add evaluation loop here


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.tokenizer)
