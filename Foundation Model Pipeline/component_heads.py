# This file handles all classification heads for each component.
# Each component (damper, insulator, etc) has its own classifier.
# The classifiers all share the same encoder but have different output layers.

import torch
import torch.nn as nn
from encoder_backbone import load_encoder
from config import ENCODER_PATH, HEAD_FILES


class DefectClassifier(nn.Module):
    def __init__(self, encoder, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        # simple MLP head on top of encoder features
        self.head = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # extract global feature (CLS token) from encoder
        feat = self.encoder.extract_global_feature(x)
        # pass through classifier head
        logits = self.head(feat)
        return logits


class HeadManager:
    def __init__(self, device="cuda"):
        self.device = device
        self.models = {}
        self.class_names = {}
        # loop through each component and load its classifier
        for component, ckpt_path in HEAD_FILES.items():
            if not ckpt_path.exists():
                continue

            ckpt = torch.load(ckpt_path, map_location=device)

            encoder = load_encoder(ENCODER_PATH, device=device)
            model = DefectClassifier(
                encoder=encoder,
                num_classes=len(ckpt["class_names"]),
                hidden_dim=256,
                dropout=0.2
            ).to(device)

            model.load_state_dict(ckpt["model"])
            model.eval()

            self.models[component] = model
            self.class_names[component] = ckpt["class_names"]

    @torch.no_grad()
    def predict(self, component_name, crop_tensor):
        # run classifier for a single cropped image
        model = self.models[component_name]
        logits = model(crop_tensor)
        # convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        # get best prediction
        conf, pred_idx = probs.max(dim=1)

        pred_idx = pred_idx.item()
        return {
            "pred_class": self.class_names[component_name][pred_idx],
            "cls_conf": conf.item(),
            "all_probs": probs.squeeze(0).cpu().tolist(),
        }