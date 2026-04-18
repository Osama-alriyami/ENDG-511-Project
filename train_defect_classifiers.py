# This script trains a classifier (task head) for each component.
# It uses a pretrained encoder (ViT/MAE) and trains a small MLP on top.
# Each component (damper, insulator, etc) gets its own classifier.

import os
import glob
import json
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image



# SETTINGS for training
# Controls dataset paths, training params, and model behavior
@dataclass
class Settings:
    # root folder containing all component datasets
    data_root: str = "taskhead_classfier_dataset"

    # pretrained encoder checkpoint
    encoder_path: str = "checkpoints_more_img/encoder_only.pt"

    # where trained classifiers will be saved
    save_root: str = "defect_classifiers_20epoch_4fine_tune"

    components: tuple = ("damper", "fitting", "insulator", "plate")

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 0

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4

    freeze_encoder: bool = True
    unfreeze_last_blocks: int = 4

    hidden_dim: int = 256
    dropout: float = 0.2

    use_weighted_sampler: bool = True
    use_class_weights: bool = False


cfg = Settings()
device = "cuda" if torch.cuda.is_available() else "cpu"


#  FUNCTIONS
def set_seed(seed_value=0):
    # fixes randomness for reproducibility
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_image_paths(folder):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_list = []

    for ext in extensions:
        image_list.extend(glob.glob(os.path.join(folder, ext)))

    image_list.sort()
    return image_list




# dataset for loading images per component
# expects structure: root/class_name/image.jpg
class ComponentDataset(Dataset):
    def __init__(self, root_folder, img_size=224):
        self.root_folder = root_folder

        self.class_names = sorted([
            # find all class folders
            name for name in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, name))
        ])

        if len(self.class_names) == 0:
            raise RuntimeError(f"No class folders found inside: {root_folder}")

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.samples = []

        for class_name in self.class_names:
            class_folder = os.path.join(root_folder, class_name)
            class_images = get_image_paths(class_folder)

            for img_path in class_images:
                self.samples.append((img_path, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {root_folder}")
        # basic augmentation to improve generalization
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label, img_path



class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            bias=qkv_bias,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x

# Vision Transformer encoder (same as used during pretraining)
# used to extract features from images
class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.enc_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat([cls_token, x], dim=1)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.enc_norm(x)
        return x

    def extract_global_feature(self, x):
        tokens = self.forward(x)
        return tokens[:, 0, :]



# small classifier head on top of encoder features
class DefectHeadModel(nn.Module):
    def __init__(self, encoder, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # get features from encoder
        features = self.encoder.extract_global_feature(x)
        # classify features
        logits = self.head(features)
        return logits



# loads pretrained encoder weights from checkpoint
def load_encoder(encoder_path, device="cuda"):
    checkpoint = torch.load(encoder_path, map_location=device)

    encoder = ViTEncoder(
        img_size=checkpoint["img_size"],
        patch_size=checkpoint["patch_size"],
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0
    ).to(device)

    encoder.patch_embed.load_state_dict(checkpoint["patch_embed"])
    encoder.cls_token.data.copy_(checkpoint["cls_token"])
    encoder.pos_embed.data.copy_(checkpoint["pos_embed"])
    encoder.encoder_blocks.load_state_dict(checkpoint["encoder_blocks"])
    encoder.enc_norm.load_state_dict(checkpoint["enc_norm"])

    return encoder



# controls which parts of encoder are trainable
# used for fine-tuning
def set_encoder_trainable(encoder, freeze_encoder=True, unfreeze_last_blocks=0):
    # freeze everything first
    for param in encoder.parameters():
        param.requires_grad = False

    if not freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = True
        return
    # optionally unfreeze last few transformer blocks
    if unfreeze_last_blocks > 0:
        for block in encoder.encoder_blocks[-unfreeze_last_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        for param in encoder.enc_norm.parameters():
            param.requires_grad = True



# IMBALANCE 
# handles class imbalance using sampling
def make_weighted_sampler(dataset):
    labels = [label for _, label in dataset.samples]

    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    weights = [1.0 / class_counts[label] for label in labels]
    weights = torch.tensor(weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler

# alternative: use loss weighting instead of sampling
def make_class_weights(dataset, device):
    labels = [label for _, label in dataset.samples]
    num_classes = len(dataset.class_names)

    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        counts[label] += 1.0

    weights = 1.0 / counts.clamp(min=1.0)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)



# EVAL
# evaluates model on validation/test set
@torch.no_grad()
def evaluate_model(model, loader, num_classes):
    model.eval()
     # build confusion matrix
    total = 0
    correct = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        total += labels.numel()
        correct += (preds == labels).sum().item()

        for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
            confusion[true_label.item(), pred_label.item()] += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, confusion

   
def show_confusion_matrix(confusion, class_names):
    print("\nConfusion Matrix")

    header = "true\\pred".ljust(16)
    for class_name in class_names:
        header += class_name[:12].ljust(14)
    print(header)

    for i, class_name in enumerate(class_names):
        row = class_name[:14].ljust(16)
        for j in range(len(class_names)):
            row += str(confusion[i, j].item()).ljust(14)
        print(row)




# trains classifier for one component
def train_one_component(component_name):
    print("\n" + "=" * 80)
    print(f"Training classifier for: {component_name}")
    print("=" * 80)
    # load train/test datasets
    train_folder = os.path.join(cfg.data_root, component_name, "train")
    test_folder = os.path.join(cfg.data_root, component_name, "test")

    if not os.path.isdir(train_folder):
        print(f"Train folder is missing: {train_folder}")
        return

    if not os.path.isdir(test_folder):
        print(f"Test folder is missing: {test_folder}")
        return

    train_dataset = ComponentDataset(train_folder, img_size=cfg.img_size)
    test_dataset = ComponentDataset(test_folder, img_size=cfg.img_size)

    if train_dataset.class_names != test_dataset.class_names:
        print("Train classes:", train_dataset.class_names)
        print("Test classes :", test_dataset.class_names)
        raise RuntimeError(f"Train and test classes do not match for {component_name}")

    if cfg.use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
        # load encoder and attach classifier head

    encoder = load_encoder(cfg.encoder_path, device=device)
    set_encoder_trainable(
        encoder,
        freeze_encoder=cfg.freeze_encoder,
        unfreeze_last_blocks=cfg.unfreeze_last_blocks
    )

    model = DefectHeadModel(
        encoder=encoder,
        num_classes=len(train_dataset.class_names),
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout
    ).to(device)

    if cfg.use_class_weights:
        class_weights = make_class_weights(train_dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        # only train parameters that are not frozen
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    best_test_acc = -1.0
    os.makedirs(cfg.save_root, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()

        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
             # forward pass
            outputs = model(images)
             # compute loss
            loss = criterion(outputs, labels)
            
             # backward + update
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            total += labels.numel()
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total if total > 0 else 0.0

        test_acc, conf = evaluate_model(model, test_loader, len(train_dataset.class_names))

        print(
            f"Epoch {epoch + 1:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f}"
        )
        # save model only if test accuracy improves
        if test_acc > best_test_acc:
            best_test_acc = test_acc

            model_save_path = os.path.join(cfg.save_root, f"{component_name}_classifier.pt")
            
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_names": train_dataset.class_names,
                    "component": component_name,
                    "img_size": cfg.img_size,
                    "freeze_encoder": cfg.freeze_encoder,
                    "unfreeze_last_blocks": cfg.unfreeze_last_blocks,
                },
                model_save_path
            )

            meta_save_path = os.path.join(cfg.save_root, f"{component_name}_classifier.json")
            with open(meta_save_path, "w") as f:
                json.dump(
                    {
                        "component": component_name,
                        "class_names": train_dataset.class_names,
                        "best_test_acc": best_test_acc,
                        "img_size": cfg.img_size,
                        "freeze_encoder": cfg.freeze_encoder,
                        "unfreeze_last_blocks": cfg.unfreeze_last_blocks,
                    },
                    f,
                    indent=2
                )

    print(f"\nBest test accuracy for {component_name}: {best_test_acc:.4f}")
    final_acc, final_conf = evaluate_model(model, test_loader, len(train_dataset.class_names))
    show_confusion_matrix(final_conf, train_dataset.class_names)



set_seed(cfg.seed)
print(f"Using device: {device}")
# train each component separately
for component_name in cfg.components:
    try:
        train_one_component(component_name)
    except Exception as error:
        print(f"\nSkipping {component_name} because of an error:")
        print(error)

