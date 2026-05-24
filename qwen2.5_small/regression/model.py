import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CoordinateAttentionCNN(nn.Module):
    """
    单个坐标的attention处理CNN分支 - 轻量化版
    """
    
    def __init__(self, 
                 in_channels: int,
                 hidden_dims: Tuple[int, ...] = (64, 128, 256),  # 轻量化网络宽度
                 dropout_rate: float = 0.3):
        """
        Args:
            in_channels: 输入通道数 (num_layers * num_heads, 现在是784)
            hidden_dims: 隐藏层维度 - 轻量化版
            dropout_rate: dropout比例
        """
        super().__init__()
        
        layers = []
        prev_channels = in_channels
        
        # 构建轻量化卷积层
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # 第一层：大幅压缩通道数，使用标准3x3卷积
                layers.extend([
                    nn.Conv2d(prev_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ])
            else:
                # 后续层：单卷积层
                layers.extend([
                    nn.Conv2d(prev_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ])
            
            # 在前两层添加MaxPool，最后一层用AdaptiveAvgPool
            if i < len(hidden_dims) - 1:
                layers.extend([
                    nn.Dropout2d(dropout_rate * 0.5),  # 2D dropout防止过拟合
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ])
            
            prev_channels = hidden_dim
        
        # 全局平均池化
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.features = nn.Sequential(*layers)
        
        # 简化的分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], 128),        # 直接映射到输出维度
        )
        
        self.output_dim = 128
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_layers*num_heads, H, W]
        
        Returns:
            features: [batch_size, output_dim]
        """
        features = self.features(x)
        output = self.classifier(features)
        return output

class AttentionBBoxRegressor(nn.Module):
    """
    基于4个坐标attention的bbox IoU回归器 - 轻量化版
    """
    
    def __init__(self,
                 num_layers: int,
                 num_heads: int = 28,                                    # 新增num_heads参数
                 hidden_dims: Tuple[int, ...] = (64, 128, 256),        # 轻量化网络宽度
                 dropout_rate: float = 0.3,
                 fusion_dim: int = 256,                                 # 轻量化融合层
                 use_sigmoid: bool = True):
        """
        Args:
            num_layers: attention的层数
            num_heads: 每层的head数量
            hidden_dims: CNN分支的隐藏层维度 - 轻量化版
            dropout_rate: dropout比例
            fusion_dim: 融合层维度 - 轻量化版
            use_sigmoid: 是否使用sigmoid将输出限制在[0,1]
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_sigmoid = use_sigmoid
        
        # 计算实际输入通道数
        actual_in_channels = num_layers * num_heads  # 28 * 28 = 784
        
        # 4个独立的CNN分支处理4个坐标
        self.x1_cnn = CoordinateAttentionCNN(actual_in_channels, hidden_dims, dropout_rate)
        self.y1_cnn = CoordinateAttentionCNN(actual_in_channels, hidden_dims, dropout_rate)
        self.x2_cnn = CoordinateAttentionCNN(actual_in_channels, hidden_dims, dropout_rate)
        self.y2_cnn = CoordinateAttentionCNN(actual_in_channels, hidden_dims, dropout_rate)
        
        # 简化的融合层
        coord_feature_dim = self.x1_cnn.output_dim
        total_feature_dim = 4 * coord_feature_dim  # 4 * 128 = 512
        
        self.fusion = nn.Sequential(
            # 第一阶段：特征融合
            nn.Linear(total_feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(dropout_rate),
            
            # 第二阶段：降维
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            # 最终输出层
            nn.Linear(fusion_dim // 2, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                x1_attention: torch.Tensor,
                y1_attention: torch.Tensor, 
                x2_attention: torch.Tensor,
                y2_attention: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1_attention: [batch_size, num_layers*num_heads, H, W]
            y1_attention: [batch_size, num_layers*num_heads, H, W]
            x2_attention: [batch_size, num_layers*num_heads, H, W]
            y2_attention: [batch_size, num_layers*num_heads, H, W]
        
        Returns:
            iou_pred: [batch_size, 1] 预测的IoU值
        """
        # 分别处理4个坐标的attention
        x1_features = self.x1_cnn(x1_attention)  # [batch_size, 128]
        y1_features = self.y1_cnn(y1_attention)  # [batch_size, 128]
        x2_features = self.x2_cnn(x2_attention)  # [batch_size, 128]
        y2_features = self.y2_cnn(y2_attention)  # [batch_size, 128]
        
        # 拼接所有特征
        combined_features = torch.cat([
            x1_features, y1_features, x2_features, y2_features
        ], dim=1)  # [batch_size, 512]
        
        # 通过融合层得到IoU预测
        iou_pred = self.fusion(combined_features)  # [batch_size, 1]
        
        # 可选择性使用sigmoid限制输出范围到[0,1]
        if self.use_sigmoid:
            iou_pred = torch.sigmoid(iou_pred)
        
        return iou_pred

class AttentionBBoxRegressorShared(nn.Module):
    """
    共享权重版本的bbox IoU回归器 - 轻量化版
    """
    
    def __init__(self,
                 num_layers: int,
                 num_heads: int = 28,                                    # 新增num_heads参数
                 hidden_dims: Tuple[int, ...] = (64, 128, 256),        # 轻量化网络宽度
                 dropout_rate: float = 0.3,
                 fusion_dim: int = 256,                                 # 轻量化融合层
                 use_sigmoid: bool = True):
        """
        Args:
            num_layers: attention的层数
            num_heads: 每层的head数量
            hidden_dims: CNN的隐藏层维度 - 轻量化版
            dropout_rate: dropout比例
            fusion_dim: 融合层维度 - 轻量化版
            use_sigmoid: 是否使用sigmoid将输出限制在[0,1]
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_sigmoid = use_sigmoid
        
        # 计算实际输入通道数
        actual_in_channels = num_layers * num_heads  # 28 * 28 = 784
        
        # 共享的CNN分支
        self.shared_cnn = CoordinateAttentionCNN(actual_in_channels, hidden_dims, dropout_rate)
        
        # 简化的融合层
        coord_feature_dim = self.shared_cnn.output_dim
        total_feature_dim = 4 * coord_feature_dim
        
        self.fusion = nn.Sequential(
            # 第一阶段：特征融合
            nn.Linear(total_feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(dropout_rate),
            
            # 第二阶段：降维
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            # 最终输出层
            nn.Linear(fusion_dim // 2, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                x1_attention: torch.Tensor,
                y1_attention: torch.Tensor, 
                x2_attention: torch.Tensor,
                y2_attention: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1_attention: [batch_size, num_layers*num_heads, H, W]
            y1_attention: [batch_size, num_layers*num_heads, H, W]
            x2_attention: [batch_size, num_layers*num_heads, H, W]
            y2_attention: [batch_size, num_layers*num_heads, H, W]
        
        Returns:
            iou_pred: [batch_size, 1] 预测的IoU值
        """
        # 使用共享CNN分别处理4个坐标的attention
        x1_features = self.shared_cnn(x1_attention)  # [batch_size, 128]
        y1_features = self.shared_cnn(y1_attention)  # [batch_size, 128]
        x2_features = self.shared_cnn(x2_attention)  # [batch_size, 128]
        y2_features = self.shared_cnn(y2_attention)  # [batch_size, 128]
        
        # 拼接所有特征
        combined_features = torch.cat([
            x1_features, y1_features, x2_features, y2_features
        ], dim=1)  # [batch_size, 512]
        
        # 通过融合层得到IoU预测
        iou_pred = self.fusion(combined_features)  # [batch_size, 1]
        
        # 可选择性使用sigmoid限制输出范围到[0,1]
        if self.use_sigmoid:
            iou_pred = torch.sigmoid(iou_pred)
        
        return iou_pred

def create_model(num_layers: int,
                num_heads: int = 28,                                    # 新增参数
                model_type: str = "independent",
                hidden_dims: Tuple[int, ...] = (64, 128, 256),        # 轻量化默认值
                dropout_rate: float = 0.3,
                fusion_dim: int = 256,                                 # 轻量化默认值
                use_sigmoid: bool = True) -> nn.Module:
    """
    创建模型的工厂函数 - 轻量化版
    
    Args:
        num_layers: attention的层数
        num_heads: 每层的head数量
        model_type: 模型类型，"independent" 或 "shared"
        hidden_dims: CNN隐藏层维度 - 轻量化版
        dropout_rate: dropout比例
        fusion_dim: 融合层维度 - 轻量化版
        use_sigmoid: 是否使用sigmoid限制输出到[0,1]
    
    Returns:
        模型实例
    """
    if model_type == "independent":
        return AttentionBBoxRegressor(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            fusion_dim=fusion_dim,
            use_sigmoid=use_sigmoid
        )
    elif model_type == "shared":
        return AttentionBBoxRegressorShared(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            fusion_dim=fusion_dim,
            use_sigmoid=use_sigmoid
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    """测试模型"""
    # 更新后的参数
    batch_size = 4
    num_layers = 28
    num_heads = 28
    height, width = 24, 24
    
    # 计算新的通道数
    channels = num_layers * num_heads  # 784
    
    # 创建测试输入
    x1_attention = torch.randn(batch_size, channels, height, width)
    y1_attention = torch.randn(batch_size, channels, height, width)
    x2_attention = torch.randn(batch_size, channels, height, width)
    y2_attention = torch.randn(batch_size, channels, height, width)
    
    print(f"Testing Lightweight AttentionBBoxRegressor...")
    print(f"Input channels: {channels} (num_layers={num_layers} × num_heads={num_heads})")
    
    model_independent = create_model(
        num_layers=num_layers, 
        num_heads=num_heads,
        model_type="independent", 
        use_sigmoid=True
    )
    model_independent.eval()
    
    with torch.no_grad():
        iou_pred = model_independent(x1_attention, y1_attention, x2_attention, y2_attention)
    
    print(f"Input shape: {x1_attention.shape}")
    print(f"Output shape: {iou_pred.shape}")
    print(f"Output range: [{iou_pred.min().item():.4f}, {iou_pred.max().item():.4f}]")
    print(f"Model parameters (independent): {count_parameters(model_independent):,}")
    
    print("\nTesting Lightweight AttentionBBoxRegressor (shared)...")
    model_shared = create_model(
        num_layers=num_layers,
        num_heads=num_heads, 
        model_type="shared", 
        use_sigmoid=True
    )
    model_shared.eval()
    
    with torch.no_grad():
        iou_pred_shared = model_shared(x1_attention, y1_attention, x2_attention, y2_attention)
    
    print(f"Output shape: {iou_pred_shared.shape}")
    print(f"Output range: [{iou_pred_shared.min().item():.4f}, {iou_pred_shared.max().item():.4f}]")
    print(f"Model parameters (shared): {count_parameters(model_shared):,}")
    
    # 测试梯度
    print("\nTesting gradient flow...")
    model_independent.train()
    iou_pred = model_independent(x1_attention, y1_attention, x2_attention, y2_attention)
    
    # 创建虚假目标并计算损失
    target_iou = torch.rand(batch_size, 1)  # 随机IoU目标值
    loss = F.mse_loss(iou_pred, target_iou)
    loss.backward()
    
    print(f"Loss (MSE): {loss.item():.6f}")
    print("Lightweight gradient flow test passed!")
    
    # 显示网络复杂度对比
    print(f"\n=== Lightweight Network Summary ===")
    print(f"Lightweight params (784 channels): {count_parameters(model_independent):,}")
    print(f"Hidden dims: {(64, 128, 256)}")
    print(f"Fusion dim: 256")
    print(f"Expected reduction: ~5-10x compared to enhanced version")

if __name__ == "__main__":
    test_model()