import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import json
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import create_dataloader
from model import create_model

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(model_path, config, device):
    """Load trained model from checkpoint"""
    # Create enhanced model with updated parameters
    model_config = config['model']
    model = create_model(
        num_layers=model_config['num_layers'],
        num_heads=model_config.get('num_heads', 28),  # NEW: support for num_heads
        model_type=model_config['type'],
        hidden_dims=tuple(model_config['hidden_dims']),
        dropout_rate=model_config['dropout_rate'],
        fusion_dim=model_config['fusion_dim'],
        use_sigmoid=model_config['use_sigmoid']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def test_model(model, dataloader, device, save_predictions=False):
    """Test model and calculate metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_metadata = []
    
    print("Running inference on test set...")
    pbar = tqdm(dataloader, desc='Testing')
    
    with torch.no_grad():
        for batch_idx, (x1_attn, y1_attn, x2_attn, y2_attn, iou_targets, metadata) in enumerate(pbar):
            # Move data to device
            x1_attn = x1_attn.to(device)
            y1_attn = y1_attn.to(device)
            x2_attn = x2_attn.to(device)
            y2_attn = y2_attn.to(device)
            iou_targets = iou_targets.to(device)
            
            # Forward pass
            iou_pred = model(x1_attn, y1_attn, x2_attn, y2_attn)
            # print(x1_attn.shape)
            # print(y1_attn.shape)
            # print(x2_attn.shape)
            # print(x2_attn.shape)
            # Store results
            all_predictions.extend(iou_pred.cpu().numpy().flatten())
            all_targets.extend(iou_targets.cpu().numpy().flatten())
            
            if save_predictions:
                # Store metadata for detailed analysis
                for i in range(len(metadata['pkl_file'])):
                    sample_meta = {
                        'pkl_file': metadata['pkl_file'][i],
                        'image_folder': metadata['image_folder'][i],
                        'response_id': metadata['response_id'][i].item(),
                        'bbox_id': metadata['bbox_id'][i].item(),
                        'pred_bbox': metadata['pred_bbox'][i],
                        'true_iou': iou_targets[i].cpu().item(),
                        'predicted_iou': iou_pred[i].cpu().item(),
                        'absolute_error': abs(iou_targets[i].cpu().item() - iou_pred[i].cpu().item())
                    }
                    all_metadata.append(sample_meta)
    
    return all_targets, all_predictions, all_metadata

def calculate_regression_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    # 基本误差指标
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # 相对误差
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # 百分位数误差
    abs_errors = np.abs(y_true - y_pred)
    metrics['mae_percentiles'] = {
        '50': np.percentile(abs_errors, 50),
        '75': np.percentile(abs_errors, 75),
        '90': np.percentile(abs_errors, 90),
        '95': np.percentile(abs_errors, 95),
        '99': np.percentile(abs_errors, 99)
    }
    
    # 相关系数
    try:
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    except:
        pearson_corr = pearson_p = 0.0
        spearman_corr = spearman_p = 0.0
    
    metrics['pearson_corr'] = pearson_corr
    metrics['pearson_p'] = pearson_p
    metrics['spearman_corr'] = spearman_corr
    metrics['spearman_p'] = spearman_p
    
    # R squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
    
    # 不同IoU范围的准确率（用于比较）
    thresholds = [0.3, 0.5, 0.7]
    for thresh in thresholds:
        true_positive = np.sum((y_true >= thresh) & (y_pred >= thresh))
        true_negative = np.sum((y_true < thresh) & (y_pred < thresh))
        total = len(y_true)
        accuracy = (true_positive + true_negative) / total
        metrics[f'threshold_accuracy_{thresh}'] = accuracy
    
    return metrics

def plot_scatter_plot(y_true, y_pred, save_path):
    """Plot scatter plot of true vs predicted IoU"""
    plt.figure(figsize=(10, 8))
    
    # 主散点图
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # 对角线（完美预测线）
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # 计算和显示R²
    pearson_corr, _ = pearsonr(y_true, y_pred)
    r2 = pearson_corr ** 2
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xlabel('True IoU')
    plt.ylabel('Predicted IoU')
    plt.title('True vs Predicted IoU - Enhanced Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residual_plot(y_true, y_pred, save_path):
    """Plot residual analysis"""
    residuals = np.array(y_pred) - np.array(y_true)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 残差 vs 预测值
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted IoU')
    axes[0, 0].set_ylabel('Residuals (Pred - True)')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差 vs 真实值
    axes[0, 1].scatter(y_true, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('True IoU')
    axes[0, 1].set_ylabel('Residuals (Pred - True)')
    axes[0, 1].set_title('Residuals vs True')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差直方图
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q图
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_analysis(y_true, y_pred, save_path):
    """Plot error analysis by IoU ranges"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    abs_errors = np.abs(y_true - y_pred)
    
    # 定义IoU区间
    iou_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_centers = [(iou_bins[i] + iou_bins[i+1]) / 2 for i in range(len(iou_bins)-1)]
    
    # 计算每个区间的统计信息
    mae_by_bin = []
    rmse_by_bin = []
    counts_by_bin = []
    
    for i in range(len(iou_bins)-1):
        mask = (y_true >= iou_bins[i]) & (y_true < iou_bins[i+1])
        if i == len(iou_bins)-2:  # 最后一个区间包含右边界
            mask = (y_true >= iou_bins[i]) & (y_true <= iou_bins[i+1])
        
        if np.sum(mask) > 0:
            mae_by_bin.append(np.mean(abs_errors[mask]))
            rmse_by_bin.append(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
            counts_by_bin.append(np.sum(mask))
        else:
            mae_by_bin.append(0)
            rmse_by_bin.append(0)
            counts_by_bin.append(0)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MAE by IoU range
    axes[0, 0].bar(range(len(mae_by_bin)), mae_by_bin, alpha=0.7)
    axes[0, 0].set_xlabel('IoU Range')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('MAE by IoU Range')
    axes[0, 0].set_xticks(range(len(mae_by_bin)))
    axes[0, 0].set_xticklabels([f'{iou_bins[i]:.1f}-{iou_bins[i+1]:.1f}' for i in range(len(iou_bins)-1)], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE by IoU range
    axes[0, 1].bar(range(len(rmse_by_bin)), rmse_by_bin, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('IoU Range')
    axes[0, 1].set_ylabel('Root Mean Square Error')
    axes[0, 1].set_title('RMSE by IoU Range')
    axes[0, 1].set_xticks(range(len(rmse_by_bin)))
    axes[0, 1].set_xticklabels([f'{iou_bins[i]:.1f}-{iou_bins[i+1]:.1f}' for i in range(len(iou_bins)-1)], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample count by IoU range
    axes[1, 0].bar(range(len(counts_by_bin)), counts_by_bin, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('IoU Range')
    axes[1, 0].set_ylabel('Sample Count')
    axes[1, 0].set_title('Sample Distribution by IoU Range')
    axes[1, 0].set_xticks(range(len(counts_by_bin)))
    axes[1, 0].set_xticklabels([f'{iou_bins[i]:.1f}-{iou_bins[i+1]:.1f}' for i in range(len(iou_bins)-1)], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error vs True IoU scatter with color coding
    axes[1, 1].scatter(y_true, abs_errors, alpha=0.6, c=abs_errors, cmap='viridis')
    axes[1, 1].set_xlabel('True IoU')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Absolute Error vs True IoU')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_predictions_by_ranges(metadata_list, save_path):
    """Analyze prediction performance across different IoU ranges"""
    if not metadata_list:
        print("Warning: No metadata available for range analysis")
        return None, None
    
    df = pd.DataFrame(metadata_list)
    
    # Define IoU bins
    iou_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df['iou_bin'] = pd.cut(df['true_iou'], bins=iou_bins, include_lowest=True)
    
    # Calculate statistics per IoU bin
    stats_by_range = df.groupby('iou_bin').agg({
        'absolute_error': ['mean', 'std', 'median', 'count'],
        'true_iou': 'mean',
        'predicted_iou': 'mean'
    }).round(4)
    
    # Flatten column names
    stats_by_range.columns = ['_'.join(col).strip() for col in stats_by_range.columns.values]
    stats_by_range = stats_by_range.reset_index()
    
    # Plot analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mean absolute error by range
    axes[0, 0].bar(range(len(stats_by_range)), stats_by_range['absolute_error_mean'])
    axes[0, 0].set_xlabel('IoU Range')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('Mean Absolute Error by IoU Range')
    axes[0, 0].set_xticks(range(len(stats_by_range)))
    axes[0, 0].set_xticklabels([str(bin_range) for bin_range in stats_by_range['iou_bin']], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample count by range
    axes[0, 1].bar(range(len(stats_by_range)), stats_by_range['absolute_error_count'], color='orange')
    axes[0, 1].set_xlabel('IoU Range')
    axes[0, 1].set_ylabel('Sample Count')
    axes[0, 1].set_title('Sample Distribution by IoU Range')
    axes[0, 1].set_xticks(range(len(stats_by_range)))
    axes[0, 1].set_xticklabels([str(bin_range) for bin_range in stats_by_range['iou_bin']], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # True vs Predicted IoU by range
    axes[1, 0].plot(range(len(stats_by_range)), stats_by_range['true_iou_mean'], 'o-', label='True IoU (mean)')
    axes[1, 0].plot(range(len(stats_by_range)), stats_by_range['predicted_iou_mean'], 's-', label='Predicted IoU (mean)')
    axes[1, 0].set_xlabel('IoU Range')
    axes[1, 0].set_ylabel('IoU Value')
    axes[1, 0].set_title('True vs Predicted IoU by Range')
    axes[1, 0].set_xticks(range(len(stats_by_range)))
    axes[1, 0].set_xticklabels([str(bin_range) for bin_range in stats_by_range['iou_bin']], rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot with error bars
    x_pos = range(len(stats_by_range))
    axes[1, 1].errorbar(x_pos, stats_by_range['absolute_error_mean'], 
                       yerr=stats_by_range['absolute_error_std'], 
                       fmt='o-', capsize=5)
    axes[1, 1].set_xlabel('IoU Range')
    axes[1, 1].set_ylabel('Absolute Error (Mean ± Std)')
    axes[1, 1].set_title('Error Statistics by IoU Range')
    axes[1, 1].set_xticks(range(len(stats_by_range)))
    axes[1, 1].set_xticklabels([str(bin_range) for bin_range in stats_by_range['iou_bin']], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_by_range, df

def save_prediction_comparison_csv(metadata_list, output_dir):
    """Save prediction comparison CSV with key fields"""
    if not metadata_list:
        print("Warning: No metadata available for CSV generation")
        return
    
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'prediction_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"Prediction comparison CSV saved to: {csv_path}")
    
    return csv_path

def main():
    parser = argparse.ArgumentParser(description='Test Enhanced BBox IoU Regressor')
    parser.add_argument('--config', type=str, default='config_regression.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save detailed predictions to CSV')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load enhanced model
    print(f"Loading enhanced model from: {args.model}")
    model, checkpoint = load_model(args.model, config, device)
    
    # Display model information
    from model import count_parameters
    model_config = config['model']
    param_count = count_parameters(model)
    input_channels = model_config['num_layers'] * model_config.get('num_heads', 28)
    
    print(f"Enhanced Model Information:")
    print(f"  - Parameters: {param_count:,}")
    print(f"  - Input channels per coordinate: {input_channels}")
    print(f"  - Hidden dimensions: {model_config['hidden_dims']}")
    print(f"  - Fusion dimension: {model_config['fusion_dim']}")
    print(f"  - Model trained for {checkpoint['epoch']} epochs")
    
    if 'metrics' in checkpoint:
        print(f"  - Training validation metrics: {checkpoint['metrics']}")
    
    # Create test dataloader
    data_config = config['data']
    target_dir = data_config['target_dir']
    root_dir = data_config['root_dir']
    # object_num = data_config['object_num']
    # base_dir = os.path.join(root_dir, f"{object_num}_object", target_dir)

    # output_dir = f"./test_results/{object_num}_object/{target_dir}"
    base_dir = os.path.join(root_dir, data_config['dataset'], target_dir)
    output_dir = f"../evaluation/regression_results/{data_config['dataset']}/{target_dir}"
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    test_loader = create_dataloader(
        csv_file=os.path.join(base_dir, data_config['test_csv']),
        pkl_dir=os.path.join(base_dir, data_config['test_pkl_dir']),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        # target_size=tuple(config['model']['target_size']),
        normalize_attention=config['model']['normalize_attention']
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Run inference
    y_true, y_pred, metadata_list = test_model(
        model, test_loader, device, save_predictions=True
    )
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    # Print results
    print(f"\nEnhanced Model Test Results:")
    print("=" * 70)
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
    print(f"R² (Coefficient of Determination): {metrics['r2']:.4f}")
    print(f"Pearson Correlation: {metrics['pearson_corr']:.4f} (p={metrics['pearson_p']:.4f})")
    print(f"Spearman Correlation: {metrics['spearman_corr']:.4f} (p={metrics['spearman_p']:.4f})")
    print()
    print("MAE Percentiles:")
    for perc, value in metrics['mae_percentiles'].items():
        print(f"  {perc}th percentile: {value:.4f}")
    print()
    print("Threshold-based Accuracy (for comparison with classification):")
    for thresh in [0.3, 0.5, 0.7]:
        acc = metrics[f'threshold_accuracy_{thresh}']
        print(f"  IoU >= {thresh}: {acc:.4f}")
    print(f"\nModel Complexity: {param_count:,} parameters")
    print(f"Input Channels: {input_channels} (28 layers × 28 heads)")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    def to_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, dict):
            return {k: to_serializable(v) for k, v in val.items()}
        if isinstance(val, list):
            return [to_serializable(v) for v in val]
        return val

    metrics_to_save = {k: to_serializable(v) for k, v in metrics.items()}
    # Add model info to metrics
    metrics_to_save['model_info'] = {
        'parameters': param_count,
        'input_channels': input_channels,
        'hidden_dims': model_config['hidden_dims'],
        'fusion_dim': model_config['fusion_dim'],
        'num_layers': model_config['num_layers'],
        'num_heads': model_config.get('num_heads', 28)
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save prediction comparison CSV
    save_prediction_comparison_csv(metadata_list, output_dir)
    
    # Plot scatter plot
    scatter_path = os.path.join(output_dir, 'scatter_plot.png')
    plot_scatter_plot(y_true, y_pred, scatter_path)
    print(f"Scatter plot saved to: {scatter_path}")
    
    # Plot residual analysis
    residual_path = os.path.join(output_dir, 'residual_analysis.png')
    plot_residual_plot(y_true, y_pred, residual_path)
    print(f"Residual analysis saved to: {residual_path}")
    
    # Plot error analysis
    error_path = os.path.join(output_dir, 'error_analysis.png')
    plot_error_analysis(y_true, y_pred, error_path)
    print(f"Error analysis saved to: {error_path}")
    
    # Analyze predictions by IoU ranges
    if metadata_list:
        range_analysis_path = os.path.join(output_dir, 'range_analysis.png')
        stats_by_range, df_detailed = analyze_predictions_by_ranges(metadata_list, range_analysis_path)
        print(f"Range analysis saved to: {range_analysis_path}")
        
        # Save detailed predictions
        detailed_path = os.path.join(output_dir, 'detailed_predictions.csv')
        df_detailed.to_csv(detailed_path, index=False)
        print(f"Detailed predictions saved to: {detailed_path}")
        
        # Save range statistics
        range_stats_path = os.path.join(output_dir, 'range_statistics.csv')
        stats_by_range.to_csv(range_stats_path, index=False)
        print(f"Range statistics saved to: {range_stats_path}")
    
    print(f"\nAll enhanced model test results saved to: {output_dir}")

if __name__ == "__main__":
    main()