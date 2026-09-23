import os
import sys
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


if len(sys.argv) < 2:
    print('Path to downloaded MiMo-V2.6-Flash-RL is required.')
    sys.exit(1)

ROOT = Path(sys.argv[1])
SRC = ROOT / "dflash"
DST = ROOT / "dflash-llama"

print('Model root:', ROOT)
print('DFlash dir:', SRC)
print('Output dir:', DST)

DST.mkdir(exist_ok=True)

# 复制 DFlash 文件
for p in SRC.iterdir():
    if p.is_file():
        shutil.copy2(p, DST / p.name)


# ---------- config ----------
cfg_path = DST / "config.json"

# 如果原文件曾有 trailing comma，请使用你之前修好的合法 JSON
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["has_embed_tokens"] = True

mask_id = cfg["dflash_config"]["mask_token_id"]

with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")


# ---------- 从主模型读取 embedding ----------
with open(ROOT / "model.safetensors.index.json", "r") as f:
    target_index = json.load(f)

embedding_key = "model.embed_tokens.weight"
embedding_file = target_index["weight_map"][embedding_key]

with safe_open(
    ROOT / embedding_file,
    framework="pt",
    device="cpu",
) as f:
    embedding = f.get_tensor(embedding_key).clone()


print("embedding:", embedding.shape, embedding.dtype)
print("mask id:", mask_id)


# ---------- 读取 MiMo 专用 mask embedding ----------
mask_obj = torch.load(
    SRC / "mask_embedding.pt",
    map_location="cpu",
    weights_only=True,
)

if isinstance(mask_obj, dict):
    if "embedding" in mask_obj:
        mask = mask_obj["embedding"]
    elif "mask_embedding" in mask_obj:
        mask = mask_obj["mask_embedding"]
    elif len(mask_obj) == 1:
        mask = next(iter(mask_obj.values()))
    else:
        raise RuntimeError(
            f"unknown mask_embedding.pt structure: {mask_obj.keys()}"
        )
else:
    mask = mask_obj

mask = mask.reshape(-1)

print("mask:", mask.shape, mask.dtype)

assert mask.numel() == embedding.shape[1]

if mask.dtype != embedding.dtype:
    mask = mask.to(embedding.dtype)


# ---------- 最关键的一步 ----------
embedding[mask_id] = mask


# DFlash converter 会把 embed_tokens.weight 映射到 model.embed_tokens.weight
emb_file = "model-mask-aware-embeddings.safetensors"

save_file(
    {"embed_tokens.weight": embedding},
    DST / emb_file,
)


# ---------- 修改 DFlash index ----------
with open(DST / "model.safetensors.index.json", "r") as f:
    index = json.load(f)

index["weight_map"]["embed_tokens.weight"] = emb_file

with open(DST / "model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)
    f.write("\n")

# ---------- 符号链接，解决找不到权重的问题 ----------
link_src = DST / "dflash_draft_model.safetensors"
link_dst = DST / "model.safetensors"
link_rel = os.path.relpath(link_src, DST)
try:
    os.symlink(link_rel, link_dst)
    print("create symlink:", link_dst, "=>", link_rel)
except FileExistsError:
    os.remove(link_dst)
    os.symlink(link_rel, link_dst)
    print("recreate symlink:", link_dst, "=>", link_rel)

print("created:", DST)
