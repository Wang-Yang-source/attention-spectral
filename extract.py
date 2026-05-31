#!/usr/bin/env python3
"""
从所有 checkpoint 中提取注意力矩阵

用法: python extract.py --ckpt_dir ./checkpoints --output attention_data.pt
"""
import torch
import os
import argparse
import re
import json
from model import TinyTransformer, TinyTransformerConfig
from train import make_probe_inputs

def load_ckpt_config(ckpt_dir):
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        return json.load(f)

def extract_all(ckpt_dir, output_path):
    cfg = load_ckpt_config(ckpt_dir)

    config = TinyTransformerConfig(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["seq_len"],
        vocab_size=cfg["vocab_size"],
    )

    device = torch.device("cpu")
    probes = make_probe_inputs(device, cfg["seq_len"])

    # 找到所有 checkpoint
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_step_") and f.endswith(".pt")],
        key=lambda f: int(re.search(r"ckpt_step_(\d+)", f).group(1))
    )

    all_data = {"probes": probes, "checkpoints": []}

    for ckpt_file in ckpt_files:
        step = int(re.search(r"ckpt_step_(\d+)", ckpt_file).group(1))
        ckpt = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location="cpu", weights_only=False)

        model = TinyTransformer(config).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        entry = {"step": step, "probe_attentions": {}}
        for probe_name, probe_input in probes.items():
            with torch.no_grad():
                _, attn = model(probe_input)
            # attn: [layer][head] -> (1, T, T)
            layers_np = []
            for layer_heads in attn:
                heads_np = torch.stack([h.squeeze(0).cpu() for h in layer_heads], dim=0).numpy()
                layers_np.append(heads_np)  # (n_heads, T, T)
            entry["probe_attentions"][probe_name] = layers_np

        # 也存一些重量信息
        entry["loss"] = ckpt["metrics"][-1]["loss"] if ckpt["metrics"] else None

        all_data["checkpoints"].append(entry)
        print(f"  Extracted step {step:6d} — loss {entry['loss']:.4f}")

    torch.save(all_data, output_path)
    print(f"\n✅ 提取完成: {len(all_data['checkpoints'])} 个 checkpoint → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint 目录")
    parser.add_argument("--output", type=str, default="attention_data.pt")
    args = parser.parse_args()
    extract_all(args.ckpt_dir, args.output)
