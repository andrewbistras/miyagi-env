# cli/train.py

import os
import hydra
import math
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorWithPadding, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from miyagi_machines import get_processed_dataset, Model, MultiObjectiveTrainer, compute_metrics

# set to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    print("Configuration loaded successfully.")
    # print(OmegaConf.to_yaml(cfg, resolve=True)) # Optional: for debugging

    # --- 1. Fetch Processed Dataset ---
    ds_dict, metadata = get_processed_dataset(
        raw_data_path=cfg.data.raw_path,
        processed_dataset_path=cfg.data.processed_dir,
        tokenizer_name=cfg.model.base_encoder,
        max_len=cfg.data.max_len,
    )
    
    print(f"Dataset loaded: {metadata['total_size']:,} total examples")
    print(f"  Splits: {metadata['splits']}")
    print(f"  Label distribution: {metadata['label_distribution']}")

    # --- 1. Calculate XE loss weights ---
    if 'label_distribution' in metadata:
        label_counts = torch.tensor(metadata['label_distribution'], dtype=torch.float)
        label_weights = label_counts.sum() / (len(label_counts) * label_counts)
        cfg.train.class_weights = label_weights.tolist()

    # For the adversarial model head
    if 'model_label_distribution' in metadata:
        model_counts = torch.tensor(metadata['model_label_distribution'], dtype=torch.float)
        model_weights = model_counts.sum() / (len(model_counts) * model_counts)
        cfg.train.model_weights = model_weights.tolist()
    
    # For the adversarial prompt head
    if 'prompt_label_distribution' in metadata:
        prompt_counts = torch.tensor(metadata['prompt_label_distribution'], dtype=torch.float)
        prompt_weights = prompt_counts.sum() / (len(prompt_counts) * prompt_counts)
        cfg.train.prompt_weights = prompt_weights.tolist()

    
    # --- 3. Instantiate Model ---
    model = Model(
        base_encoder=cfg.model.base_encoder,
        n_models=metadata['n_models'],
        n_prompts=metadata['n_prompts'],
        grl_lambda=cfg.model.grl_lambda,
    )
    
    print(f"Model instantiated with {metadata['n_models']} models and {metadata['n_prompts']} prompts.")
    print(f"Model architecture:\n{model}")


    # --- 4. Compile Model (Optional but Recommended) ---
    if torch.cuda.is_available():
        print("Compiling model for performance...")
        model = torch.compile(model, mode="default", fullgraph=True, dynamic=True)
        torch.backends.cuda.matmul.allow_tf32 = True

    # --- 5. Set Up Training Arguments ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_encoder)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    effective_batch_size = cfg.train.batch_size * cfg.train.gradient_accumulation
    steps_per_epoch = math.ceil(len(ds_dict["train"]) / effective_batch_size)
    eval_and_save_steps = math.ceil(steps_per_epoch / 2)
    
    training_args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation,
        eval_strategy="steps",
        eval_steps=eval_and_save_steps,  
        save_strategy="steps",
        save_steps=eval_and_save_steps, 
        load_best_model_at_end=True,
        metric_for_best_model="accuracy", # Changed to accuracy
        greater_is_better=True, # Accuracy is better when greater
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=False,
        dataloader_drop_last=True,
        report_to="wandb",
        logging_steps=50,
        remove_unused_columns=False,
    )

    # --- 6. Set Up Optimizer and Scheduler ---
    enc_params  = [p for n, p in model.named_parameters() if "encoder" in n]
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    
    optimizer = AdamW(
        [
            {"params": enc_params, "lr": cfg.train.lr_encoder},
            {"params": head_params, "lr": cfg.train.lr_head},
        ],
        weight_decay=0.01,
        fused=torch.cuda.is_available(),
    )
    
    num_training_steps = steps_per_epoch * cfg.train.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )
    
    # --- 7. Instantiate and Run Trainer ---
    trainer = MultiObjectiveTrainer(
        training_config=cfg.train,
        model=model,
        args=training_args,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["valid"],
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
    )
    
    print("Training started.")
    trainer.train()
    print("Training finished.")

if __name__ == "__main__":
    main()
