# /src/miyagi_machines/models/model.py

import torch.nn as nn
from transformers import AutoModel
from miyagi_machines.custom_ops import GradientReversalLayer

class Model(nn.Module):
    def __init__(self, base_encoder: str, n_models: int, n_prompts: int, grl_lambda: float = 1.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            base_encoder
        )
        self.project = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )

        self.head_main = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
        self._grl = GradientReversalLayer(alpha=grl_lambda)
        self.head_model  = nn.Linear(256, n_models)
        self.head_prompt = nn.Linear(256, n_prompts)


    def forward(self, input_ids, attention_mask, **kwargs):

        h = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        z = self.project(h)
        pseudo_z = self._grl(z)

        logits_main   = self.head_main(z)
        logits_model  = self.head_model(pseudo_z)
        logits_prompt = self.head_prompt(pseudo_z)

        return {
            "z": z,
            "logits": logits_main,      
            "logits_model": logits_model,
            "logits_prompt": logits_prompt,
        }
