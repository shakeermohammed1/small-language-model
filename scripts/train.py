import yaml
import argparse
from pathlib import Path

from src.training.model import SLM
from src.training.trainer import Trainer
from src.data.dataset import TextDataset
from src.data.preprocess import clean_text
from src.data.tokenizer import load_tokenizer, encode_corpus
from src.utils.helpers import set_seed, count_parameters, ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__, "logs/train.log")


def load_txt(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [clean_text(line) for line in f if line.strip()]


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])
    ensure_dir(cfg["paths"]["checkpoints"])

    model = SLM(**cfg["model"])
    logger.info(f"Model parameters: {count_parameters(model):,}")

    tokenizer = load_tokenizer("gpt2")
    raw_dir = Path(cfg["paths"]["data_raw"])

    train_ids = encode_corpus(tokenizer, load_txt(raw_dir / "train.txt"))
    eval_ids = encode_corpus(tokenizer, load_txt(raw_dir / "validation.txt"))
    logger.info(f"Train tokens: {len(train_ids):,} | Eval tokens: {len(eval_ids):,}")

    train_dataset = TextDataset(train_ids, cfg["model"]["block_size"])
    eval_dataset = TextDataset(eval_ids, cfg["model"]["block_size"])

    trainer = Trainer(model, train_dataset, eval_dataset, cfg["training"])

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = trainer.train_epoch()
        eval_loss = trainer.evaluate()
        logger.info(f"Epoch {epoch+1} | train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f}")

        ckpt_path = f"{cfg['paths']['checkpoints']}/epoch_{epoch+1}.pt"
        trainer.save_checkpoint(ckpt_path, epoch)
        logger.info(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args.config)
