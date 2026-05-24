import os
import re
import json
import yaml
import datetime
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dataset import create_dataloader


# ------------------------------
# 工具函数
# ------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def calculate_entropy_single(attention_map):
    """计算单个attention map的熵"""
    attention_flat = attention_map.flatten()
    attention_norm = attention_flat / (attention_flat.sum() + 1e-12)
    entropy = -np.sum(attention_norm * np.log(attention_norm + 1e-12))
    return entropy


# ------------------------------
# 主逻辑
# ------------------------------
def analyze_entropy_for_testset(batch_data, head_selection, num_heads=28):
    """对一个batch的样本计算熵并返回结果
    
    Args:
        batch_data: 批量数据
        head_selection: 每个坐标选择的(layer, head)对，格式如:
                       {"x1": [(23, 14), (12, 15)], "y1": [(20, 10)], ...}
        num_heads: 每层的头数
    """
    x1_batch, y1_batch, x2_batch, y2_batch, iou_batch, metadata = batch_data
    print("x1_batch is:", x1_batch.shape)
    x1_np, y1_np, x2_np, y2_np = (
        x1_batch.cpu().numpy(),
        y1_batch.cpu().numpy(),
        x2_batch.cpu().numpy(),
        y2_batch.cpu().numpy(),
    )
    

    B = x1_np.shape[0]
    coord_data = {"x1": x1_np, "y1": y1_np, "x2": x2_np, "y2": y2_np}
    results = []

    for i in range(B):
        sample = {}
        sample["pkl_file"] = metadata["pkl_file"][i]
        sample["image_folder"] = metadata["image_folder"][i]
        sample["response_id"] = int(metadata["response_id"][i])
        sample["bbox_id"] = int(metadata["bbox_id"][i])
        sample["pred_bbox"] = metadata["pred_bbox"][i]
        sample["iou"] = float(iou_batch[i].cpu().numpy()) if torch.is_tensor(iou_batch[i]) else float(iou_batch[i])

        entropy_values = {}
        for coord, attn in coord_data.items():
            # print(f"coord is {coord}")
            head_pairs = head_selection.get(coord, [])
            # print(f"head_pairs are {head_pairs}")
            if not head_pairs:
                continue

            coord_entropies = []
            for layer_idx, head_idx in head_pairs:
                # 计算特定head在attention数组中的位置
                attention_index = layer_idx * num_heads + head_idx
                # print(f"layer index is {layer_idx}")
                # print(f"head index is {head_idx}")
                # print(f"attention_index is {attention_index}")

                # 检查索引是否有效
                if attention_index >= attn.shape[1]:
                    print(f"Warning: Invalid index for {coord}: layer {layer_idx}, head {head_idx}")
                    continue
                
                # 提取特定head的attention map
                specific_head_attn = attn[i, attention_index, :, :]
                
                # 计算该head的熵
                entropy = calculate_entropy_single(specific_head_attn)
                coord_entropies.append(entropy)

            # 对该坐标选中的所有heads的熵取平均
            entropy_values[coord] = np.mean(coord_entropies) if coord_entropies else np.nan

        sample["x1_entropy"] = entropy_values.get("x1", np.nan)
        sample["y1_entropy"] = entropy_values.get("y1", np.nan)
        sample["x2_entropy"] = entropy_values.get("x2", np.nan)
        sample["y2_entropy"] = entropy_values.get("y2", np.nan)

        # 计算所有坐标的平均熵
        entropies = [v for v in entropy_values.values() if not np.isnan(v)]
        sample["mean_entropy"] = float(np.mean(entropies)) if entropies else np.nan

        results.append(sample)

    return results


# ------------------------------
# 主程序入口
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Select bounding box by entropy (head-level selection)")
    parser.add_argument("--config", type=str, default="config_regression.yaml")
    parser.add_argument("--save_dir", type=str, default="./entropy_selection_results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument(
        "--head_selection",
        type=str,
        default='{"x1":[[23,14],[12,15]],"y1":[[20,10]],"x2":[[19,5]],"y2":[[14,7]]}',
        help="Dictionary string specifying selected (layer, head) pairs per coordinate"
    )
    parser.add_argument(
        "--selection_mode", 
        type=str, 
        default="value",
        choices=["value", "rank"],
        help="Selection mode: 'value' for mean entropy, 'rank' for rank sum"
    )
    args = parser.parse_args()

    head_selection_raw = json.loads(args.head_selection.replace("'", '"'))
    head_selection = {}
    for coord, pairs in head_selection_raw.items():
        head_selection[coord] = [tuple(pair) for pair in pairs]
    
    print(f"Head selection per coordinate:")
    for coord, heads in head_selection.items():
        print(f"  {coord}: {heads}")
    print(f"Selection mode: {args.selection_mode}")

    # 加载配置文件
    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    # 输出目录
    save_dir = os.path.join(args.save_dir, data_cfg['target_dir'])
    ensure_dir(save_dir)

    # 创建 test dataloader
    base_dir = os.path.join(
        data_cfg["root_dir"],
        data_cfg["dataset"],
        data_cfg["target_dir"],
    )

    test_loader = create_dataloader(
        csv_file=os.path.join(base_dir, data_cfg["test_csv"]),
        pkl_dir=os.path.join(base_dir, data_cfg["test_pkl_dir"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        normalize_attention=cfg["model"]["normalize_attention"],
    )

    print(f"Loaded {len(test_loader.dataset)} test samples.")

    # 遍历测试集并计算熵
    all_samples = []
    for batch in tqdm(test_loader, desc="Computing entropy"):
        batch_results = analyze_entropy_for_testset(batch, head_selection, args.num_heads)
        all_samples.extend(batch_results)

    df_all = pd.DataFrame(all_samples)
    print(f"Total bbox entries processed: {len(df_all)}")

    # CSV 输出目录
    csv_output_dir = os.path.join(save_dir, "individual_csvs")
    ensure_dir(csv_output_dir)

    grouped = df_all.groupby("image_folder")
    all_selected = []

    for image_folder, group in grouped:
        if args.selection_mode == "value":
            # Value模式：按mean_entropy排序 (stable: tie-break 跟随输入 pkl_file 字母序)
            group_sorted = group.sort_values(by="mean_entropy", ascending=True, kind="stable").copy()
            
            # 保存CSV
            sub_csv_path = os.path.join(csv_output_dir, f"{image_folder}_value.csv")
            group_sorted.to_csv(sub_csv_path, index=False)
            
            # 选择mean_entropy最小的
            best_row = group_sorted.iloc[0]
            
        else:  # rank mode
            # Rank模式：计算每个坐标的rank
            group_copy = group.copy()
            
            # 计算每个坐标的rank（升序，最小值rank=1）
            group_copy["x1_rank"] = group_copy["x1_entropy"].rank(method='min', na_option='bottom')
            group_copy["y1_rank"] = group_copy["y1_entropy"].rank(method='min', na_option='bottom')
            group_copy["x2_rank"] = group_copy["x2_entropy"].rank(method='min', na_option='bottom')
            group_copy["y2_rank"] = group_copy["y2_entropy"].rank(method='min', na_option='bottom')
            
            # 计算rank_sum
            group_copy["rank_sum"] = (
                group_copy["x1_rank"] + 
                group_copy["y1_rank"] + 
                group_copy["x2_rank"] + 
                group_copy["y2_rank"]
            )
            
            # 按rank_sum排序 (default quicksort — try to match _check which used unstable)
            group_sorted = group_copy.sort_values(by="rank_sum", ascending=True)
            
            # 保存CSV（包含rank信息）
            sub_csv_path = os.path.join(csv_output_dir, f"{image_folder}_rank.csv")
            group_sorted.to_csv(sub_csv_path, index=False)
            
            # 选择rank_sum最小的
            best_row = group_sorted.iloc[0]

        # 从pkl文件名中提取 "response_n.pkl"
        pkl_name = os.path.basename(best_row["pkl_file"])
        match = re.search(r"response_(\d+)", pkl_name)
        short_name = f"response_{match.group(1)}.pkl" if match else pkl_name

        all_selected.append({
            "subdir": image_folder,
            "pkl_path": short_name,
            "bbox_id": int(best_row["bbox_id"])
        })

    # 汇总最终选择结果
    df_selected = pd.DataFrame(all_selected, columns=["subdir", "pkl_path", "bbox_id"])
    final_csv = os.path.join(save_dir, f"entropy_{data_cfg['target_dir']}.csv")
    df_selected.to_csv(final_csv, index=False)

    print(f"\n✅ Final selected bbox CSV saved: {final_csv}")
    print(f"Total selected images: {len(df_selected)}")


if __name__ == "__main__":
    main()