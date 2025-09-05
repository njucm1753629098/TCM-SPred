import torch

# Data paths
DATA_PATHS = {
    'positive': 'data/D1.herb_effective_symptom.xlsx',
    'negative': 'data/D2.herb_ineffective_symptom.xlsx',
    'herb_symbol': 'data/D3.herb_target.xlsx',
    'symptom_symbol': 'data/D4.symptom_symbol.xlsx',
    'ppi': 'data/D5.combine_score.tsv',
    'word2vec': 'word2vec_models/tcm_word2vec.model'
}

# Training settings
TRAIN_CONFIG = {
    'batch_size': 8,
    'epochs': 50,
    'lr': 5e-5,
    'weight_decay': 1e-3,
    'train_split': 0.7,
    'val_split': 0.15
}

# Prediction settings
PREDICT_CONFIG = {
    'output_folder': "predictions",
    'matrix_output': "symptom-herb_matrix.csv",
    'heatmap_output': "symptom-herb_heatmap.pdf"
}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_model.pth"
METRICS_SAVE_PATH = "metrics_data.pth"