"""Create deterministic masked LM documents and paired reward examples for each split."""

from pathlib import Path

import numpy as np
import yaml

from megatron.data.indexed_dataset import MMapIndexedDatasetBuilder


def prepare(directory, seq_length):
    directory.mkdir(exist_ok=True)
    for seed, split in enumerate(("train", "valid", "test"), start=100):
        rng = np.random.default_rng(seed)
        # Include targets from both halves of the 512-entry vocabulary.
        pos = rng.integers(2, 512, (64, seq_length + 1), dtype=np.int32)
        neg = rng.integers(2, 512, pos.shape, dtype=np.int32)
        lm_labels = pos.copy()
        lm_labels[:, 1::3] = -100
        arrays = {"pos": pos, "neg": neg, "lm_labels": lm_labels}
        for side, docs in (("pos", pos), ("neg", neg)):
            labels = np.full_like(docs, -100)
            for i, doc in enumerate(docs):
                index = rng.integers(1, seq_length)
                labels[i, index] = doc[index]
            arrays[f"{side}_labels"] = labels
        for name, docs in arrays.items():
            prefix = directory / f"{split}_{name}"
            builder = MMapIndexedDatasetBuilder(str(prefix) + ".bin", dtype=np.int32)
            for doc in docs:
                builder.add_item(doc)
                builder.end_document()
            builder.finalize(str(prefix) + ".idx")


if __name__ == "__main__":
    base = yaml.safe_load((Path(__file__).parent / "configs/base.yml").read_text())
    prepare(Path("data"), base["seq_length"])
