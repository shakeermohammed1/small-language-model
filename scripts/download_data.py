from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm


def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    print("Downloading roneneldan/TinyStories from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories")

    for split in ["train", "validation"]:
        out_path = Path("data/raw") / f"{split}.txt"
        print(f"Writing {len(ds[split]):,} stories to {out_path} ...")
        with open(out_path, "w", encoding="utf-8") as f:
            for example in tqdm(ds[split], desc=split):
                text = example["text"].strip()
                if text:
                    f.write(text + "\n")
        print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
