# Vision Transformer encoder used for feature extraction.
# Based on MAE-style encoder (no decoder here).
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoderOnly(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.enc_norm(x)
        return x

    def extract_global_feature(self, x):
        tokens = self.forward(x)
        # return CLS token as global representation
        return tokens[:, 0, :]


def load_encoder(encoder_path, device="cuda"):
    ckpt = torch.load(encoder_path, map_location=device)

    model = ViTEncoderOnly(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0
    ).to(device)

    # if checkpoint was saved with wrapper dict
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model