import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from model import initialize_model
from data_loader import load_data, generate_samples, load_word2vec_model
from utils import HerbSymptomDataset, dynamic_collate_fn
from train_eval import train_model, evaluate
from config import TRAIN_CONFIG, DEVICE, MODEL_SAVE_PATH
import os
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import pickle
import warnings
warnings.filterwarnings("ignore")

# ------------------ utils ------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_plotting():
    sns.set(style="white")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 30,
        'legend.fontsize': 16,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (16, 8)
    })

COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b'   # burgundy
]

# ------------------ persist ------------------
RESULT_FILE = "fold_results.pkl"

def save_fold_results(results, path=RESULT_FILE):
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Fold results saved to {path}")

def load_fold_results(path=RESULT_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        results = pickle.load(f)
    print(f"Fold results loaded from {path}")
    return results

# ------------------ cross-validation ------------------
def run_cross_validation(n_splits=10):
    set_seed(42)
    print("Loading data...")
    df_pos, df_neg, df_herb, df_symptom, ppi_dict = load_data()
    all_samples = generate_samples(df_pos, df_neg, df_herb, df_symptom, ppi_dict)
    word2vec_model = load_word2vec_model()
    dataset = HerbSymptomDataset(all_samples, word2vec_model)

    print(f"Total samples: {len(dataset)}")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{n_splits}")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)

        val_size = int(TRAIN_CONFIG['val_split'] * len(train_subset))
        train_size = len(train_subset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_subset, [train_size, val_size]
        )

        train_loader = DataLoader(train_subset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, collate_fn=dynamic_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=TRAIN_CONFIG['batch_size'], collate_fn=dynamic_collate_fn)
        test_loader = DataLoader(test_subset, batch_size=TRAIN_CONFIG['batch_size'], collate_fn=dynamic_collate_fn)

        model = initialize_model()
        print(f"Training fold {fold + 1}...")
        train_model(model, train_loader, val_loader, DEVICE, TRAIN_CONFIG['epochs'])

        print(f"Evaluating fold {fold + 1}...")
        _, auc_score, _ = evaluate(model, test_loader, DEVICE)

        all_labels, all_probs = [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Collecting predictions"):
                inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(DEVICE)

                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        print(f"Fold {fold + 1} AUC: {roc_auc:.4f}")

        fold_results.append({
            'fold': fold + 1,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'test_labels': all_labels,
            'test_probs': all_probs
        })

    save_fold_results(fold_results)
    return fold_results

# ------------------ plot ------------------
# ------------------ plot ------------------
def plot_roc_curves(fold_results, save_path="cross_validation_roc.pdf"):
    setup_plotting()
    plt.figure(figsize=(10, 8))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    color_cycle = COLORS[:len(fold_results)]

    for i, result in enumerate(fold_results):
        fpr = result['fpr']
        tpr = result['tpr']
        roc_auc = result['auc']

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, linestyle='--', lw=1.5, alpha=0.8,
                 color=color_cycle[i % len(color_cycle)],
                 label=f'Fold {i + 1} (AUC = {roc_auc:.4f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color=COLORS[-1], linestyle='-', lw=2.5,
             alpha=1.0, label=f'Mean ROC (AUC = {mean_auc:.4f})')

    
    # plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    # plt.grid(True, alpha=0.3)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{len(fold_results)}-Fold Cross-Validation ROC Curves', fontweight='bold', fontsize=22)
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"ROC plot saved to {save_path}")
    plt.show()
    return mean_auc

# ------------------ main ------------------
def main(n_splits=10):
    fold_results = load_fold_results()
    if fold_results is None or len(fold_results) != n_splits:
        fold_results = run_cross_validation(n_splits=n_splits)
    else:
        print("Skip training, use cached results.")

    mean_auc = plot_roc_curves(fold_results)
    print(f"\nCross-validation done. Mean AUC: {mean_auc:.4f}")

if __name__ == "__main__":
    main(n_splits=10)