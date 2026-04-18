import os
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import ENCODER_PATH, PROTONET_FILES, TRAIN_DATA_ROOT, IMG_SIZE, DEVICE
from encoder_backbone import load_encoder


def list_images(root):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    root = str(root)
    for e in exts:
        paths.extend(glob.glob(os.path.join(root, e)))
    paths.sort()
    return paths


class ProtoNetModel(nn.Module):
    def __init__(self, encoder, hidden_dim=256, emb_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        feat = self.encoder.extract_global_feature(x)
        z = self.proj(feat)
        z = F.normalize(z, dim=1)
        return z


class ProtoNetHeadManager:
    def __init__(self, device=DEVICE):
        self.device = device
        self.models = {}
        self.class_names = {}
        self.prototypes = {}

        self.tfm = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

        shared_encoder = load_encoder(ENCODER_PATH, device=device)

        for component, ckpt_path in PROTONET_FILES.items():
            if not ckpt_path.exists():
                continue

            ckpt = torch.load(ckpt_path, map_location=device)

            model = ProtoNetModel(
                encoder=shared_encoder,
                hidden_dim=ckpt["hidden_dim"],
                emb_dim=ckpt["emb_dim"],
                dropout=ckpt["dropout"]
            ).to(device)

            model.load_state_dict(ckpt["model"], strict=False)
            model.eval()

            class_names = ckpt["class_names"]
            prototypes = self._build_prototypes(model, component, class_names)

            self.models[component] = model
            self.class_names[component] = class_names
            self.prototypes[component] = prototypes

    @torch.no_grad()
    def _embed_pil(self, model, image):
        x = self.tfm(image).unsqueeze(0).to(self.device)
        z = model(x)
        return z

    @torch.no_grad()
    def _build_prototypes(self, model, component, class_names):
        train_root = Path(TRAIN_DATA_ROOT) / component / "train"

        all_embs = []
        all_labels = []

        for class_idx, class_name in enumerate(class_names):
            class_dir = train_root / class_name
            paths = list_images(class_dir)

            if len(paths) == 0:
                continue

            for p in paths:
                img = Image.open(p).convert("RGB")
                z = self._embed_pil(model, img)
                all_embs.append(z.squeeze(0))
                all_labels.append(class_idx)

        if len(all_embs) == 0:
            raise RuntimeError(f"No prototype images found for component: {component}")

        all_embs = torch.stack(all_embs, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)

        prototypes = []
        for class_idx in range(len(class_names)):
            cls_embs = all_embs[all_labels == class_idx]
            if cls_embs.shape[0] == 0:
                proto = torch.zeros(all_embs.shape[1], device=self.device)
            else:
                proto = cls_embs.mean(dim=0)
            proto = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
            prototypes.append(proto)

        return torch.stack(prototypes, dim=0)

    @torch.no_grad()
    def predict(self, component_name, crop_tensor, topk=3):
        model = self.models[component_name]
        class_names = self.class_names[component_name]
        prototypes = self.prototypes[component_name]

        z = model(crop_tensor)
        dists = torch.cdist(z, prototypes)
        logits = -dists
        probs = torch.softmax(logits, dim=1)

        conf, pred_idx = probs.max(dim=1)
        pred_idx = pred_idx.item()
        conf = conf.item()

        probs_1d = probs.squeeze(0)
        k = min(topk, len(class_names))
        top_vals, top_idx = torch.topk(probs_1d, k=k)

        topk_preds = []
        for rank in range(k):
            idx = top_idx[rank].item()
            val = top_vals[rank].item()
            topk_preds.append((class_names[idx], val))

        return {
            "pred_class": class_names[pred_idx],
            "cls_conf": conf,
            "all_probs": probs_1d.cpu().tolist(),
            "topk_preds": topk_preds,
        }