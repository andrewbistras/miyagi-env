# /src/miyagi_machines/datasets/dataset.py

import os
import json
from collections import Counter
from datasets import Dataset, DatasetDict, ClassLabel, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

# --- Helper functions (mostly unchanged) ---

def load_raw_dataset(jsonl_path):
    """Load JSONL memory-efficiently using a generator."""
    print(f"Loading raw data from: {jsonl_path}")
    def _generator():
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line {i+1}: {e}")
                        continue
    return Dataset.from_generator(_generator)

def prepare_labels(ds):
    """Cast string model -> integer ClassLabel columns."""

    ai_names = sorted([g for g in ds.unique("model") if g != "human"])
    name2id  = {name: i for i, name in enumerate(ai_names)}

    if "labels" not in ds.column_names: ds = ds.map(lambda ex: {"labels": 0 if ex["model"]=="human" else 1})
    ds = ds.remove_columns(["model", "lang"])
    
    prompt_names = sorted(ds.unique("source"))
    prompt_feat  = ClassLabel(names=prompt_names)
    ds = ds.cast_column("source", prompt_feat)
    ds = ds.rename_column("source", "labels_prompt")
    return ds

def oversample_split(ds_split):
    """Oversamples a given dataset split (e.g., the training set)."""
    counts = Counter(ds_split["labels"])
    major_label, minor_label = max(counts, key=counts.get), min(counts, key=counts.get)
    needed = counts[major_label] - counts[minor_label]

    if needed > 0:
        print(f"Oversampling minority label '{minor_label}', adding {needed} examples.")
        minority_ds = ds_split.filter(lambda ex: ex["labels"] == minor_label, num_proc=1).shuffle(seed=42)
        oversampled_ds = concatenate_datasets([ds_split, minority_ds.select(range(needed))]).shuffle(seed=42)
        return oversampled_ds
    return ds_split

def get_model_counts(ds):
    """Return the number of models and prompts for model initialization."""
    n_models  = len([v for v in ds.unique("labels_model") if v >= 0])
    n_prompts = ds.features["labels_prompt"].num_classes
    return n_models, n_prompts

def stratified_split(ds, test_frac=0.05, seed=42):
    """Stratified train/valid/test split."""
    split = ds.train_test_split(test_size=test_frac, seed=seed, stratify_by_column="labels")
    tmp   = split["train"].train_test_split(test_size=test_frac, seed=seed*2, stratify_by_column="labels")
    return DatasetDict(train=tmp["train"], valid=tmp["test"], test=split["test"])


# --- Sole API call ---

def get_processed_dataset(
    raw_data_path,
    processed_dataset_path,
    tokenizer_name,
    max_len,
    test_frac=0.05,
    seed=42
):
    """
    Main function to fetch the dataset.
    If a pre-processed version exists at `processed_dataset_path`, it's loaded.
    Otherwise, it creates, processes, tokenizes, and saves the dataset.
    """
    if os.path.exists(processed_dataset_path):
        print(f"Loading pre-processed dataset from {processed_dataset_path}...")
        processed_ds = load_from_disk(processed_dataset_path)
        metadata = {
            "n_models": len([v for v in processed_ds["train"].unique("labels_model") if v >= 0]),
            "n_prompts": processed_ds["train"].features["labels_prompt"].num_classes
        }
        return processed_ds, metadata

    print("No pre-processed dataset found. Creating from scratch...")
    
    # 1. Load and Label
    raw = load_raw_dataset(raw_data_path)
    labeled_ds = prepare_labels(raw)

    # 2. Get model counts for metadata BEFORE splitting/sampling
    n_models, n_prompts = get_model_counts(labeled_ds)
    metadata = {"n_models": n_models, "n_prompts": n_prompts}

    # 3. Split the data
    ds_dict = stratified_split(labeled_ds, test_frac, seed)

    # 4. Oversample ONLY the training set
    ds_dict["train"] = oversample_split(ds_dict["train"])

    # 5. Tokenize all splits
    print(f"Tokenizing splits with tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    for split in ds_dict.keys():
        ds_dict[split] = ds_dict[split].map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"] # text is no longer needed after tokenization
        )
    
    # 6. Set final torch format for the trainer
    cols_to_keep = [
        "input_ids", "attention_mask", "labels", "labels_model", "labels_prompt"
    ]
    ds_dict.set_format(type="torch", columns=cols_to_keep)
    
    # 7. Save to disk for future runs
    print(f"Saving processed dataset to {processed_dataset_path}...")
    ds_dict.save_to_disk(processed_dataset_path)

    return ds_dict, metadata