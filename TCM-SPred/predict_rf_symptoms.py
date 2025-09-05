import os
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
from predict_utils import predict_symptom
from data_loader import get_embedding, load_word2vec_model
from model import initialize_model
from config import DEVICE, MODEL_SAVE_PATH, PREDICT_CONFIG

# set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict_rf_symptoms(seed=42):
    set_seed(seed)
    rf_symptoms = [
        '浮肿', '乏力', '腹痛', '腹胀', '呼吸困难', '咳嗽', '纳差', '呕吐',
        '皮肤瘙痒', '四肢痛', '头痛', '头晕', '胃脘嘈杂', '消瘦', '心慌',
        '胸痛', '多汗', '多尿', '烦躁', '关节不利', '关节痛', '汗出',
        '活动不利', '气促', '情绪不稳', '身痛', '下肢无力', '血尿',
        '厌食', '易怒'
    ]
    print(f"Renal fibrosis symptoms to predict: {rf_symptoms}")

    output_folder = os.path.join(PREDICT_CONFIG['output_folder'], "Renal_fibrosis_prediction")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Results saved to: {output_folder}")

    model = initialize_model()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {MODEL_SAVE_PATH}")

    word2vec_model = load_word2vec_model()

    print("Loading herb list...")
    _, _, df_herb, _, _ = load_data()
    all_herbs = list(df_herb['herb'].unique())
    print(f"Found {len(all_herbs)} herbs")

    symptom_results = {}
    device = next(model.parameters()).device

    for symptom in rf_symptoms:
        print(f"\nPredicting: {symptom}")
        symptom_df = predict_symptom(symptom, model, word2vec_model, all_herbs, device)

        output_file = os.path.join(output_folder, f"Renal_fibrosis_{symptom}_scores.csv")
        symptom_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Saved: {output_file}")

        symptom_results[symptom] = symptom_df.set_index('herb')['score']

    print("\nGenerating total-score ranking...")
    combined_df = pd.DataFrame(symptom_results)
    combined_df['total_score'] = combined_df.sum(axis=1, skipna=True)
    combined_df = combined_df.sort_values('total_score', ascending=False)

    total_output_file = os.path.join(output_folder, "Renal_fibrosis_total_ranking.csv")
    combined_df.to_csv(total_output_file, encoding='utf-8-sig')
    print(f"Total-score ranking saved to: {total_output_file}")

    print("\nTop 10 herbs by total score:")
    for i, (herb, row) in enumerate(combined_df.head(10).iterrows(), 1):
        print(f"{i}. {herb}: {row['total_score']:.4f}")

def load_data():
    import pandas as pd
    from config import DATA_PATHS
    df_herb = pd.read_excel(DATA_PATHS['herb_symbol'])
    return None, None, df_herb, None, None

if __name__ == "__main__":
    predict_rf_symptoms(seed=42)