import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from config import PREDICT_CONFIG
from data_loader import get_embedding

def run_prediction(model, word2vec_model, all_symptoms, all_herbs):
    """run full prediction pipeline"""
    output_folder = PREDICT_CONFIG['output_folder']
    os.makedirs(output_folder, exist_ok=True)

    device = next(model.parameters()).device
    matrix_data = {}

    for symptom in tqdm(all_symptoms, desc="predicting symptoms"):
        symptom_df = predict_symptom(symptom, model, word2vec_model, all_herbs, device)

        output_file = os.path.join(output_folder, f"{symptom}_herb_scores.csv")
        symptom_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        matrix_data[symptom] = symptom_df.set_index('herb')['score'].to_dict()

    matrix_df = pd.DataFrame(matrix_data).T
    matrix_df.index.name = 'symptom'

    matrix_output = os.path.join(output_folder, PREDICT_CONFIG['matrix_output'])
    matrix_df.to_csv(matrix_output, encoding='utf-8-sig')
    print(f"\nsymptom-herb matrix saved to: {matrix_output}")

    return matrix_df

def predict_symptom(symptom, model, word2vec_model, all_herbs, device):
    """predict scores for one symptom across all herbs"""
    symptom_embed = get_embedding(symptom, word2vec_model)
    herb_scores = []

    batch_size = 64
    for i in tqdm(range(0, len(all_herbs), batch_size),
                  desc=f"predicting '{symptom}'", leave=False):
        batch_herbs = all_herbs[i:i + batch_size]

        herb_embeds = [get_embedding(h, word2vec_model) for h in batch_herbs]
        herb_tensors = torch.tensor(np.array(herb_embeds), dtype=torch.float32).to(device)

        batch = {
            'herb': herb_tensors,
            'symptom': torch.tensor(symptom_embed).repeat(len(batch_herbs), 1).to(device),
            'herb_symbols': torch.zeros(len(batch_herbs), 1, 129, dtype=torch.float32).to(device),
            'herb_lengths': torch.ones(len(batch_herbs), dtype=torch.long).to(device),
            'symptom_symbols': torch.zeros(len(batch_herbs), 1, 129, dtype=torch.float32).to(device),
            'symptom_lengths': torch.ones(len(batch_herbs), dtype=torch.long).to(device)
        }

        with torch.no_grad():
            outputs = model(batch)
            scores = torch.sigmoid(outputs).cpu().numpy().flatten()

        for herb, score in zip(batch_herbs, scores):
            herb_scores.append((herb, score))

    df = pd.DataFrame(herb_scores, columns=['herb', 'score'])
    df = df.sort_values('score', ascending=False)

    return df