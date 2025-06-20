# /train.py

import os
import yaml
import math
import argparse
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

def main():
    # --- 1. Load Configuration from YAML ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['train']

    print("Configuration loaded successfully.")

    # --- 2. Fetch Processed Dataset ---
    ds_dict, metadata = get_processed_dataset(
        raw_data_path=data_cfg['raw_path'],
        processed_dataset_path=data_cfg['processed_dir'],
        tokenizer_name=model_cfg['base_encoder'],
        max_len=data_cfg['max_len'],
    )
    
    print(f"Dataset loaded: {metadata['total_size']:,} total examples")
    print(f"  Splits: {metadata['splits']}")
    print(f"  Label distribution: {metadata['label_distribution']}")

    
    # --- 3. Instantiate Model ---
    model = Model(
        base_encoder=model_cfg['base_encoder'],
        n_models=metadata['n_models'],
        n_prompts=metadata['n_prompts'],
        grl_lambda=model_cfg['grl_lambda'],
    )

    # --- 4. Compile Model (Optional but Recommended) ---
    if torch.cuda.is_available():
        print("Compiling model for performance...")
        model = torch.compile(model, mode="default", fullgraph=True, dynamic=True)
        torch.backends.cuda.matmul.allow_tf32 = True

    # --- 5. Set Up Training Arguments ---
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['base_encoder'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    effective_batch_size = train_cfg['batch_size'] * train_cfg['gradient_accumulation']
    steps_per_epoch = math.ceil(len(ds_dict["train"]) / effective_batch_size)
    eval_and_save_steps = math.ceil(steps_per_epoch / 2)
    
    training_args = TrainingArguments(
        output_dir=train_cfg['output_dir'],
        num_train_epochs=train_cfg['epochs'],
        per_device_train_batch_size=train_cfg['batch_size'],
        per_device_eval_batch_size=train_cfg['batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation'],
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
            {"params": enc_params, "lr": train_cfg['lr_encoder']},
            {"params": head_params, "lr": train_cfg['lr_head']},
        ],
        weight_decay=0.01,
        fused=torch.cuda.is_available(),
    )
    
    num_training_steps = steps_per_epoch * train_cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )
    
    # --- 7. Instantiate and Run Trainer ---
    trainer = MultiObjectiveTrainer(
        training_config=train_cfg,
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
