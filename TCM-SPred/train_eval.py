import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_recall_curve, auc, precision_score, recall_score, f1_score
)
from config import DEVICE, MODEL_SAVE_PATH, METRICS_SAVE_PATH, TRAIN_CONFIG
from utils import save_metrics, load_metrics
import os
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors   # 放在 import matplotlib.pyplot as plt 后面即可
# Set global seed for reproducibility
SEED = 42

def set_seed(seed=SEED):
    """Set all random seeds for full reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic CUDA algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Apply seed at module import
set_seed()

# Plotting setup
def setup_plotting():
    """Configure matplotlib and seaborn styles"""
    sns.set(style="white")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'legend.fontsize': 16,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (12, 8)
    })

COLOR_PALETTE = {
    'blue1': '#1f77b4', 'blue2': '#4c8cb5', 'blue3': '#7aa1c2',
    'orange1': '#ff7f0e', 'orange2': '#ff9e4a', 'orange3': '#ffbd7f',
    'purple': '#9467bd', 'red': '#d62728', 'green': '#2ca02c'
}

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

# --- Evaluation & Training ---
def evaluate(model, loader, device, criterion=None):
    """Evaluate model on validation/test set"""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auc_score = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, [1 if p[0] > 0.5 else 0 for p in all_preds])
    return total_loss / len(loader) if criterion else None, auc_score, accuracy

def train_model(model, train_loader, val_loader, device, epochs=50):
    """Train model with full reproducibility"""
    # Re-apply seed at start of training
    set_seed()
    
    # Initialize loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=TRAIN_CONFIG['lr'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Configure learning rate scheduler
    warmup_epochs = 5
    scheduler1 = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    scheduler2 = CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs, 
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler1, scheduler2], 
        milestones=[warmup_epochs]
    )
    
    # Initialize tracking variables
    train_losses, train_accs, val_losses, val_aucs, val_accs = [], [], [], [], []
    best_auc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss, train_preds, train_labels = 0, [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, [1 if p[0] > 0.5 else 0 for p in train_preds])
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        # Validation evaluation
        val_loss, val_auc, val_acc = evaluate(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_accs.append(val_acc)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc:.4f}")

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save final model
    final_model_path = "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return train_losses, train_accs, val_losses, val_aucs, val_accs

def test_model(model, test_loader, device):
    """Evaluate model on test set"""
    # Load best model for testing
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array([1 if p[0] > 0.5 else 0 for p in all_preds])

    test_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'conf_matrix': confusion_matrix(y_true, y_pred),
        'pred_probs': [p[0] for p in all_preds],
        'true_labels': y_true
    }

    # Class distribution
    pos_count = sum(y_true)
    class_distribution = {
        'Positive': int(pos_count),
        'Negative': int(len(y_true) - pos_count)
    }
    
    return test_metrics, class_distribution

# --- Plotting Functions ---
def plot_loss_curve(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curve"""
    setup_plotting()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train', lw=2, color=COLOR_PALETTE['blue1'])
    plt.plot(val_losses, label='Validation', lw=2, color=COLOR_PALETTE['orange1'], linestyle='--')
    plt.title('Training and Validation Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_accuracy_curve(train_accs, val_accs, save_path=None):
    """Plot training and validation accuracy curve"""
    setup_plotting()
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train', lw=2, color=COLOR_PALETTE['green'])
    plt.plot(val_accs, label='Validation', lw=2, color=COLOR_PALETTE['red'], linestyle='--')
    plt.title('Training and Validation Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_key_metrics(test_metrics, save_path=None):
    """Plot key performance metrics bar chart"""
    setup_plotting()
    plt.figure(figsize=(10, 6))
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [test_metrics['precision'], test_metrics['recall'], test_metrics['f1']]
    plt.bar(metrics, values, color=[
        COLOR_PALETTE['purple'], 
        COLOR_PALETTE['orange1'], 
        COLOR_PALETTE['green']
    ], edgecolor='w', linewidth=2)
    for i, v in enumerate(values): 
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12)
    plt.ylim(0, 1.1)
    plt.title('Key Performance Metrics', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_pr_curve(test_metrics, save_path=None):
    """Plot precision-recall curve"""
    setup_plotting()
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(
        test_metrics['true_labels'], 
        test_metrics['pred_probs']
    )
    pr_auc_val = auc(recall, precision)
    plt.plot(recall, precision, color=COLOR_PALETTE['purple'], lw=3,
             label=f'PR Curve (AUC = {pr_auc_val:.3f})')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_accuracy_donut(test_metrics, save_path=None):
    """Plot test accuracy as donut chart"""
    setup_plotting()
    plt.figure(figsize=(8, 8))
    acc = test_metrics['accuracy']
    plt.pie([acc, 1 - acc], colors=[COLOR_PALETTE['blue1'], 'lightgrey'],
            startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))
    plt.text(0, 0, f'{acc:.1%}', ha='center', va='center', fontsize=24, color=COLOR_PALETTE['blue1'])
    plt.title('Test Accuracy', fontweight='bold')
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()



import matplotlib.colors as mcolors   # 顶部已存在即可

import matplotlib.patches as mcolors  # 顶部已存在即可
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def plot_class_distribution(test_metrics, class_distribution, save_path=None):
    """Plot class distribution and confusion matrix"""
    setup_plotting()
    plt.figure(figsize=(12, 8))

    if class_distribution and 'conf_matrix' in test_metrics:
        cm = test_metrics['conf_matrix']
        cats = ['Actual\nPositive', 'TP', 'FN',
                'Actual\nNegative', 'TN', 'FP']
        vals = [class_distribution['Positive'], cm[1, 1], cm[1, 0],
                class_distribution['Negative'], cm[0, 0], cm[0, 1]]

        # 颜色
        blue_color  = COLOR_PALETTE['blue1']
        orange_color = COLOR_PALETTE['orange1']
        cols = [blue_color, COLOR_PALETTE['blue2'], COLOR_PALETTE['blue3'],
                orange_color, COLOR_PALETTE['orange2'], COLOR_PALETTE['orange3']]

        # 纹理
        hatch_list = ['///', '///', '///', '\\\\\\', '\\\\\\', '\\\\\\']

        # 画柱子
        bars = plt.bar(
            cats, vals,
            color='white',
            edgecolor=cols,
            linewidth=1.4,
            hatch=hatch_list
        )

        # 设置纹理颜色
        for bar, c in zip(bars, cols):
            bar._hatch_color = mcolors.to_rgb(c)

        # 数值标签
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h, f'{int(h)}',
                     ha='center', va='bottom', fontsize=18, color='black')

        # 图例：边框颜色分别为 blue1 和 orange1
        legend_handles = [
            mpatches.Rectangle((0, 0), 1, 1, facecolor='white',
                               edgecolor=blue_color, linewidth=1.2,
                               hatch='///', label='Positive Class'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor='white',
                               edgecolor=orange_color, linewidth=1.2,
                               hatch='\\\\\\', label='Negative Class')
        ]
        plt.legend(handles=legend_handles, loc='upper right', fontsize=14)

        plt.title('Class Distribution & Predictions', fontweight='bold')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    
# --- Combined Plotting ---
def plot_all_metrics(train_losses, val_losses, train_accs, val_accs,
                     test_metrics, class_distribution, save_folder="metrics_plots"):
    """Generate and save all evaluation plots"""
    import os
    os.makedirs(save_folder, exist_ok=True)
    plot_loss_curve(train_losses, val_losses, os.path.join(save_folder, "loss_curve.pdf"))
    plot_accuracy_curve(train_accs, val_accs, os.path.join(save_folder, "accuracy_curve.pdf"))
    plot_key_metrics(test_metrics, os.path.join(save_folder, "key_metrics.pdf"))
    plot_pr_curve(test_metrics, os.path.join(save_folder, "pr_curve.pdf"))
    plot_accuracy_donut(test_metrics, os.path.join(save_folder, "accuracy_donut.pdf"))
    plot_class_distribution(test_metrics, class_distribution,
                            os.path.join(save_folder, "class_distribution.pdf"))
    print(f"All plots saved to {save_folder}")

def plot_from_saved_metrics(metrics_path=METRICS_SAVE_PATH, save_folder="metrics_plots"):
    """Generate plots from saved metrics data"""
    (train_losses, train_accs, val_losses, val_aucs, val_accs,
     test_metrics, class_distribution) = load_metrics(metrics_path)
    plot_all_metrics(train_losses, val_losses, train_accs, val_accs,
                     test_metrics, class_distribution, save_folder)