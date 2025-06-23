import os
from collections import Counter
from datasets import (
    DatasetDict, load_from_disk, load_dataset,
    Features, ClassLabel, Value, Sequence
)
from transformers import AutoTokenizer


def oversample_split(ds_split, seed=42):
    """Balance the minority class by random oversampling."""
    counts = Counter(ds_split["labels"])
    if len(counts) < 2:
        return ds_split
    major, minor = max(counts, key=counts.get), min(counts, key=counts.get)
    needed = counts[major] - counts[minor]
    if needed <= 0:
        return ds_split
    print(f"Oversampling minority '{minor}' by {needed} examples")
    minor_ds = ds_split.filter(lambda ex: ex["labels"] == minor).shuffle(seed=seed)
    return ds_split.concatenate(minor_ds.select(range(needed))).shuffle(seed=seed)


def get_processed_dataset(
    raw_path: str,
    cache_dir: str,
    tokenizer_name: str,
    max_len: int,
    test_frac: float = 0.05,
    seed: int = 42,
) -> (DatasetDict, dict):
    # 0) quick load if cached
    if os.path.isdir(cache_dir):
        ds_dict = load_from_disk(cache_dir)
        return ds_dict, ds_dict.info.metadata

    # 1) load raw JSONL dataset
    ds = load_dataset("json", data_files=raw_path, split="train", keep_in_memory=True)

    # 2) discover unique classes (including human)
    model_names = sorted(ds.unique("model"))
    prompt_names = sorted(ds.unique("source"))

    # 3) build mapping dicts
    bin_map = {name: int(name != "human") for name in model_names}
    model2id = {name: idx for idx, name in enumerate(model_names)}
    prompt2id = {name: idx for idx, name in enumerate(prompt_names)}

    # 4) map labels and tokenize in one pass
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    def preprocess(batch):
        labels = [bin_map[m] for m in batch["model"]]
        lm = [model2id[m] for m in batch["model"]]
        lp = [prompt2id[s] for s in batch["source"]]
        tokens = tok(batch["text"], truncation=True, max_length=max_len)
        return {**tokens, "labels": labels, "labels_model": lm, "labels_prompt": lp}

    ds = ds.map(preprocess, batched=True, batch_size=500,
                 remove_columns=["text", "model", "source"])

    # 5) split and oversample
    split = ds.train_test_split(test_size=test_frac, seed=seed, stratify_by_column="labels")
    tmp = split["train"].train_test_split(test_size=test_frac, seed=seed*2, stratify_by_column="labels")
    train_ds = oversample_split(tmp["train"], seed)
    ds_dict = DatasetDict(train=train_ds, valid=tmp["test"], test=split["test"])

    # 6) cast all label fields to ClassLabel / int
    feature_schema = Features({
        "labels":       ClassLabel(names=["human", "ai"]),
        "labels_model": ClassLabel(names=model_names),
        "labels_prompt":ClassLabel(names=prompt_names),
        "input_ids":      Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
    })
    for split_name in ds_dict:
        ds_dict[split_name] = ds_dict[split_name].cast(feature_schema)
        ds_dict[split_name].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels", "labels_model", "labels_prompt"],
        )

    # 7) metadata including direct class lists
    metadata = {
        "models":  model_names,
        "prompts": prompt_names,
        "binary": ["human", "ai"],
    }
    ds_dict.info.metadata = metadata

    # 8) save and return
    ds_dict.save_to_disk(cache_dir)
    return ds_dict, metadata
