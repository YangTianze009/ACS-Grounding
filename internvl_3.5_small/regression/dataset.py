import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import ast

class AttentionBBoxDataset(Dataset):
    """
    用于bbox IoU回归的attention数据集
    固定使用16x16的grid size，不进行resize
    """
    
    def __init__(self, 
                 csv_file: str, 
                 pkl_dir: str,
                 normalize_attention: bool = True):
        """
        Args:
            csv_file: CSV文件路径，包含样本清单
            pkl_dir: pkl文件所在目录
            normalize_attention: 是否对attention进行归一化
        """
        self.csv_file = csv_file
        self.pkl_dir = pkl_dir
        self.normalize_attention = normalize_attention
        self.grid_size = (16, 16)  # 固定使用16x16
        
        # 读取CSV文件
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} samples from {csv_file}")
        
        # 打印IoU数据分布统计
        iou_stats = self.df['iou'].describe()
        print(f"IoU statistics:")
        print(f"  Mean: {iou_stats['mean']:.4f}")
        print(f"  Std: {iou_stats['std']:.4f}")
        print(f"  Min: {iou_stats['min']:.4f}")
        print(f"  Max: {iou_stats['max']:.4f}")
        print(f"  Median: {iou_stats['50%']:.4f}")
        
        # 打印IoU范围分布
        iou_ranges = [
            (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
        ]
        print(f"IoU distribution by ranges:")
        for low, high in iou_ranges:
            count = len(self.df[(self.df['iou'] >= low) & (self.df['iou'] < high)])
            print(f"  [{low:.1f}, {high:.1f}): {count} samples")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                           torch.Tensor, torch.Tensor, Dict]:
        """
        返回一个样本
        
        Returns:
            x1_attention: [num_layers * num_heads, 16, 16]
            y1_attention: [num_layers * num_heads, 16, 16] 
            x2_attention: [num_layers * num_heads, 16, 16]
            y2_attention: [num_layers * num_heads, 16, 16]
            iou_target: [1] IoU值tensor (回归目标)
            metadata: 包含额外信息的字典
        """
        row = self.df.iloc[idx]
        pkl_file = row['pkl_file']
        pkl_path = os.path.join(self.pkl_dir, pkl_file)
        
        # 加载pkl文件
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading {pkl_path}: {e}")
        
        # 提取attention数据
        attention_data = data['attention_data']
        bbox_info = data['bbox_info'] 
        metadata = data['metadata']
        
        # 获取原始grid形状并验证
        original_height, original_width = metadata['original_grid_shape']
        
        # 验证grid size必须是16x16
        assert (original_height, original_width) == self.grid_size, \
            f"Expected grid size {self.grid_size}, but got ({original_height}, {original_width}) in {pkl_file}"
        
        # 处理四个坐标的attention
        processed_attentions = {}
        
        for coord in ['x1', 'y1', 'x2', 'y2']:
            attention_key = f'{coord}_attention_map'
            attention_map = attention_data[attention_key]  # [num_layers, num_heads, vision_tokens]
            
            # 验证形状
            if len(attention_map.shape) == 3:
                num_layers, num_heads, vision_tokens = attention_map.shape
                new_channels = num_layers * num_heads
                
                # 验证vision_tokens数量
                expected_tokens = original_height * original_width
                if vision_tokens != expected_tokens:
                    raise ValueError(f"Attention shape mismatch for {coord}: "
                                   f"expected {expected_tokens}, got {vision_tokens}")
                
                # Step 1: 重新组织维度 [num_layers * num_heads, vision_tokens]
                reshaped_attention = attention_map.reshape(new_channels, vision_tokens)
                
                # Step 2: reshape到2D grid [num_layers * num_heads, 16, 16]
                attention_2d = reshaped_attention.reshape(new_channels, original_height, original_width)
                
            elif len(attention_map.shape) == 2:
                # 兼容旧格式 [num_layers, vision_tokens]
                print(f"Warning: Found 2D attention format for {coord}, treating as [num_layers, vision_tokens]")
                num_layers, vision_tokens = attention_map.shape
                new_channels = num_layers
                
                expected_tokens = original_height * original_width
                if vision_tokens != expected_tokens:
                    raise ValueError(f"Attention shape mismatch for {coord}: "
                                   f"expected {expected_tokens}, got {vision_tokens}")
                
                # 直接reshape到2D grid [num_layers, 16, 16]
                attention_2d = attention_map.reshape(new_channels, original_height, original_width)
            else:
                raise ValueError(f"Unexpected attention shape for {coord}: {attention_map.shape}")
            
            # 转换为torch tensor
            attention_tensor = torch.tensor(attention_2d, dtype=torch.float32)
            
            # 归一化（可选）
            if self.normalize_attention:
                # 对每个channel分别进行归一化
                for channel_idx in range(attention_tensor.shape[0]):
                    channel_attention = attention_tensor[channel_idx]
                    channel_min = channel_attention.min()
                    channel_max = channel_attention.max()
                    if channel_max > channel_min:
                        attention_tensor[channel_idx] = (channel_attention - channel_min) / (channel_max - channel_min)
                    else:
                        # 如果全是相同值，设为0
                        attention_tensor[channel_idx] = torch.zeros_like(channel_attention)
            
            processed_attentions[coord] = attention_tensor
        
        # 获取IoU作为回归目标
        iou_target = torch.tensor([bbox_info['iou']], dtype=torch.float32)
        
        # 构建metadata字典
        sample_metadata = {
            'pkl_file': pkl_file,
            'image_folder': bbox_info['image_folder'],
            'response_id': bbox_info['response_id'],
            'bbox_id': bbox_info['bbox_id'],
            'pred_bbox': str(bbox_info['pred_bbox']),
            'iou': bbox_info['iou'],
            'original_grid_shape': (original_height, original_width),
            'grid_size': self.grid_size,
            'verification_passed': metadata['verification_passed'],
            'attention_channels': processed_attentions['x1'].shape[0]  # 记录实际channel数
        }
        
        return (processed_attentions['x1'], 
                processed_attentions['y1'],
                processed_attentions['x2'], 
                processed_attentions['y2'],
                iou_target,
                sample_metadata)

def create_dataloader(csv_file: str, 
                     pkl_dir: str,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     normalize_attention: bool = True) -> DataLoader:
    """
    创建DataLoader
    
    Args:
        csv_file: CSV文件路径
        pkl_dir: pkl文件目录
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数
        normalize_attention: 是否归一化attention
    
    Returns:
        DataLoader对象
    """
    dataset = AttentionBBoxDataset(
        csv_file=csv_file,
        pkl_dir=pkl_dir,
        normalize_attention=normalize_attention
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader

def test_dataset():
    """测试数据集加载"""
    # 测试路径
    base_dir = "processed_data/temp_1.0_top_p_0.9_response_10"
    
    csv_file = os.path.join(base_dir, "train_list.csv")
    pkl_dir = os.path.join(base_dir, "train")
    
    # 创建数据集
    dataset = AttentionBBoxDataset(csv_file, pkl_dir)
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    x1_attn, y1_attn, x2_attn, y2_attn, iou_target, metadata = sample
    
    print(f"x1_attention shape: {x1_attn.shape}")
    print(f"y1_attention shape: {y1_attn.shape}")
    print(f"x2_attention shape: {x2_attn.shape}")
    print(f"y2_attention shape: {y2_attn.shape}")
    print(f"IoU Target: {iou_target}")
    print(f"Metadata keys: {list(metadata.keys())}")
    print(f"IoU: {metadata['iou']}")
    print(f"Grid size: {metadata['grid_size']}")
    print(f"Attention channels: {metadata['attention_channels']}")
    
    # 测试DataLoader
    dataloader = create_dataloader(csv_file, pkl_dir, batch_size=4, num_workers=0)
    
    print(f"\nTesting DataLoader...")
    for batch_idx, (x1_batch, y1_batch, x2_batch, y2_batch, iou_batch, metadata_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  x1_batch shape: {x1_batch.shape}")
        print(f"  y1_batch shape: {y1_batch.shape}")
        print(f"  x2_batch shape: {x2_batch.shape}")
        print(f"  y2_batch shape: {y2_batch.shape}")
        print(f"  iou_batch shape: {iou_batch.shape}")
        print(f"  IoU values: {iou_batch.squeeze().tolist()}")
        
        if batch_idx >= 2:  # 只测试前3个batch
            break
    
    print("Dataset test completed!")

if __name__ == "__main__":
    test_dataset()