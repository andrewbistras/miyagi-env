# /src/miyagi_machines/trainer.py

import torch
import torch.nn as nn
from transformers import Trainer
from miyagi_machines.custom_ops import SupConLoss

class MultiObjectiveTrainer(Trainer):
    """
    Custom Trainer to handle the multi-objective loss calculation.
    It reads its specific hyperparameters from a dictionary.
    """
    def __init__(self, training_config, **kwargs):
        super().__init__(**kwargs)
        self.config = training_config
        # Instantiate loss functions with parameters from the config
        self.supcon = SupConLoss(temperature=self.config['scl_temperature'])

        cls_weights = torch.tensor(self.config.get('class_weights'), device=self.args.device) if self.config.get('class_weights') else None
        self.cls_loss = nn.CrossEntropyLoss(weight=cls_weights)
        model_weights = torch.tensor(self.config.get('model_weights'), device=self.args.device) if self.config.get('model_weights') else None
        self.model_loss = nn.CrossEntropyLoss(weight=model_weights, ignore_index=-100)
        prompt_weights = torch.tensor(self.config.get('prompt_weights'), device=self.args.device) if self.config.get('prompt_weights') else None
        self.prompt_loss = nn.CrossEntropyLoss(weight=prompt_weights)

        self.ramp_steps = self.config['ramp_steps']

    def compute_loss(self, model, inputs, return_outputs=False):
        # The dataloader provides all necessary keys. Pop them for loss calculation.
        labels_cls = inputs.pop("labels")
        labels_model = inputs.pop("labels_model")
        labels_prompt = inputs.pop("labels_prompt")

        # Forward pass through the model
        outputs = model(**inputs)

        # Calculate individual loss components
        loss_cls = self.cls_loss(outputs["logits"], labels_cls)
        loss_model = self.model_loss(outputs["logits_model"], labels_model)
        loss_prompt = self.prompt_loss(outputs["logits_prompt"], labels_prompt)
        loss_scl = self.supcon(outputs["z"], labels_cls)

        # Linearly scale the adversarial loss during ramp-up phase
        global_step = self.state.global_step
        adv_scale = min(1.0, global_step / self.ramp_steps)
        
        # Combine losses using weights from the config
        loss = (
            loss_cls
            + adv_scale * self.config['grl_lambda'] * loss_model
            + adv_scale * self.config['grl_lambda'] * loss_prompt
            + self.config['w_scl'] * loss_scl
        )

        # Update outputs with detached losses for logging
        outputs.update({
            "loss_cls":   loss_cls.detach(),
            "loss_model": loss_model.detach(),
            "loss_prompt":loss_prompt.detach(),
            "loss_scl":  loss_scl.detach(),
        })

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, *args):
        # Custom prediction_step to send all outputs to compute_metrics
        loss, out, labels = super().prediction_step(model, inputs, prediction_loss_only=False)
        
        # Ensure all required outputs are present for compute_metrics
        out_cpu = {k: v.detach().cpu() for k, v in out.items()}
        labels_cpu = labels.detach().cpu()
        
        return loss.detach().cpu(), out_cpu, labels_cpu


def compute_metrics(eval_pred):
    """Calculates accuracy and average loss components during evaluation."""
    preds_dict, labels = eval_pred
    preds = torch.argmax(torch.from_numpy(preds_dict["logits"]), dim=-1)
    acc   = (preds == torch.from_numpy(labels)).float().mean()

    return {
        "accuracy"         : acc.item(),
        "loss_cls_eval"    : preds_dict["loss_cls"].mean().item(),
        "loss_model_eval"  : preds_dict["loss_model"].mean().item(),
        "loss_prompt_eval" : preds_dict["loss_prompt"].mean().item(),
        "loss_scl_eval"    : preds_dict["loss_scl"].mean().item(),
    }
