import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json
from scipy.stats import pearsonr, spearmanr

from dataset import create_dataloader
from model import create_model, count_parameters

class EarlyStopping:
    """Early stopping utility for regression"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        # 对于回归，我们希望MAE越小越好
        score = -val_score  # 转换为越大越好
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_loss_function(loss_type):
    """Create loss function based on config"""
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 基本误差指标
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # 相对误差
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # 相关系数
    try:
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    except:
        pearson_corr = pearson_p = 0.0
        spearman_corr = spearman_p = 0.0
    
    # R squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'r2': r2
    }

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    
    for batch_idx, (x1_attn, y1_attn, x2_attn, y2_attn, iou_targets, metadata) in enumerate(pbar):
        # Move data to device
        x1_attn = x1_attn.to(device)
        y1_attn = y1_attn.to(device)
        x2_attn = x2_attn.to(device)
        y2_attn = y2_attn.to(device)
        iou_targets = iou_targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        iou_pred = model(x1_attn, y1_attn, x2_attn, y2_attn)
        
        # Calculate loss
        loss = criterion(iou_pred, iou_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        all_predictions.extend(iou_pred.detach().cpu().numpy().flatten())
        all_targets.extend(iou_targets.detach().cpu().numpy().flatten())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
        
        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_regression_metrics(all_targets, all_predictions)
    metrics['loss'] = avg_loss
    
    return metrics

def validate_epoch(model, dataloader, criterion, device, epoch, writer=None):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Validation')
    
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
            
            # Calculate loss
            loss = criterion(iou_pred, iou_targets)
            
            # Accumulate metrics
            total_loss += loss.item()
            all_predictions.extend(iou_pred.detach().cpu().numpy().flatten())
            all_targets.extend(iou_targets.detach().cpu().numpy().flatten())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_regression_metrics(all_targets, all_predictions)
    metrics['loss'] = avg_loss
    
    return metrics

def create_optimizer(model, config):
    """Create optimizer based on config"""
    opt_config = config['optimizer']
    
    if opt_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif opt_config['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=opt_config['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    return optimizer

def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config"""
    sched_config = config['scheduler']
    
    if sched_config['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['T_max']
        )
    elif sched_config['type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    elif sched_config['type'] == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {sched_config['type']}")
    
    return scheduler

def save_model(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced BBox IoU Regressor')
    parser.add_argument('--config', type=str, default='config_regression.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{config['logging']['experiment_name']}_{timestamp}"
    save_dir = os.path.join(os.path.join(config['logging']['save_dir'], config['data']['target_dir']), experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config to experiment directory
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Setup tensorboard writer
    writer = SummaryWriter(save_dir)
    
    # Create data loaders
    data_config = config['data']
    target_dir = data_config['target_dir']
    root_dir = data_config['root_dir']
    dataset_name = data_config["dataset"]
    object_num = data_config["object_num"]
    base_dir = os.path.join(root_dir, dataset_name, f"{object_num}_object", target_dir)

    train_loader = create_dataloader(
        csv_file=os.path.join(base_dir, data_config['train_csv']),
        pkl_dir=os.path.join(base_dir, data_config['train_pkl_dir']),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['dataloader']['num_workers'],
        target_size=tuple(config['model']['target_size']),
        normalize_attention=config['model']['normalize_attention']
    )
    
    val_loader = create_dataloader(
        csv_file=os.path.join(base_dir, data_config['val_csv']),
        pkl_dir=os.path.join(base_dir, data_config['val_pkl_dir']),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        target_size=tuple(config['model']['target_size']),
        normalize_attention=config['model']['normalize_attention']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create enhanced model with new parameters
    model_config = config['model']
    model = create_model(
        num_layers=model_config['num_layers'],
        num_heads=model_config.get('num_heads', 28),  # NEW: with default fallback
        model_type=model_config['type'],
        hidden_dims=tuple(model_config['hidden_dims']),
        dropout_rate=model_config['dropout_rate'],
        fusion_dim=model_config['fusion_dim'],
        use_sigmoid=model_config['use_sigmoid']
    ).to(device)
    
    print(f"Enhanced model parameters: {count_parameters(model):,}")
    print(f"Input channels per coordinate: {model_config['num_layers'] * model_config.get('num_heads', 28)}")
    print(f"Hidden dimensions: {model_config['hidden_dims']}")
    print(f"Fusion dimension: {model_config['fusion_dim']}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Setup loss function
    criterion = create_loss_function(config['training']['loss_type'])
    print(f"Using loss function: {config['training']['loss_type']}")
    
    # Setup early stopping (based on MAE)
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {args.resume}, epoch {start_epoch}")
    
    # Training loop
    best_val_mae = float('inf')
    train_history = []
    val_history = []
    
    print(f"\n🚀 Starting training with enhanced network...")
    print(f"📊 Network capacity: {count_parameters(model):,} parameters")
    print(f"🔧 Batch size: {config['training']['batch_size']} (adjusted for larger model)")
    print(f"📈 Learning rate: {config['training']['learning_rate']} (adjusted for larger model)")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, writer)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log metrics
        for key in train_metrics:
            writer.add_scalar(f'Train/{key}', train_metrics[key], epoch)
        for key in val_metrics:
            writer.add_scalar(f'Val/{key}', val_metrics[key], epoch)
        
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, "
              f"RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, "
              f"RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        print(f"Val   - Pearson: {val_metrics['pearson_corr']:.4f}")
        
        # Save metrics
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        
        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            if config['logging']['save_best_model']:
                save_model(model, optimizer, epoch, val_metrics, 
                          os.path.join(save_dir, 'best_model.pth'))
                print(f"✅ Saved best model with validation MAE: {best_val_mae:.4f}")
        
        # Early stopping (based on MAE)
        if early_stopping(val_metrics['mae'], model):
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    if config['logging']['save_last_model']:
        save_model(model, optimizer, epoch, val_metrics, 
                  os.path.join(save_dir, 'last_model.pth'))
    
    # Save training history
    history = {
        'train': train_history,
        'val': val_history
    }
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    writer.close()
    print(f"\n🎉 Training completed! Results saved to: {save_dir}")
    print(f"🏆 Best validation MAE: {best_val_mae:.4f}")
    print(f"💾 Final model parameters: {count_parameters(model):,}")

if __name__ == "__main__":
    main()