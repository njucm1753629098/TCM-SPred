import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from config import DATA_PATHS, TRAIN_CONFIG
from tqdm import tqdm

def load_data():
    """load all data files and optimize large PPI loading"""
    df_positive = pd.read_excel(DATA_PATHS['positive'])
    df_negative = pd.read_excel(DATA_PATHS['negative'])
    df_herb_symbol = pd.read_excel(DATA_PATHS['herb_symbol'])
    df_symptom_symbol = pd.read_excel(DATA_PATHS['symptom_symbol'])

    assert not df_herb_symbol['herb'].isnull().any(), "empty herb names"
    assert not df_symptom_symbol['Symptom'].isnull().any(), "empty symptom names"

    ppi_path = DATA_PATHS['ppi']
    ppi_dict = load_ppi_data(ppi_path)

    return df_positive, df_negative, df_herb_symbol, df_symptom_symbol, ppi_dict

def load_ppi_data(ppi_path):
    """load large PPI file, support TSV/Excel"""
    ppi_dict = {}
    file_ext = os.path.splitext(ppi_path)[1].lower()

    if file_ext == '.tsv':
        print(f"loading large TSV: {ppi_path}")
        chunksize = 10**6
        total_rows = 0
        with open(ppi_path, 'r') as f:
            total_rows = sum(1 for _ in f) - 1
        print(f"total rows: {total_rows:,}")

        for chunk in pd.read_csv(ppi_path, sep='\t', chunksize=chunksize):
            for _, row in chunk.iterrows():
                gene1, gene2, score = row['Gene1'], row['Gene2'], row['combine_score']
                ppi_dict[(gene1, gene2)] = score
                ppi_dict[(gene2, gene1)] = score

    elif file_ext in ['.xlsx', '.xls']:
        print(f"loading Excel: {ppi_path}")
        df_ppi = pd.read_excel(ppi_path)
        for _, row in df_ppi.iterrows():
            gene1, gene2, score = row['Gene1'], row['Gene2'], row['combine_score']
            ppi_dict[(gene1, gene2)] = score
            ppi_dict[(gene2, gene1)] = score

    else:
        raise ValueError(f"unsupported PPI format: {file_ext}")

    print(f"PPI dict ready, {len(ppi_dict)//2} relations")
    return ppi_dict

def generate_samples(df_pos, df_neg, df_herb, df_symptom, ppi_dict):
    """generate training samples"""
    samples = []

    symptom_gene_map = df_symptom.groupby('Symptom')['symbol'].apply(list).to_dict()
    herb_target_map = df_herb.groupby('herb')['target'].apply(list).to_dict()

    print("processing positive samples...")
    for (herb, symptom), _ in tqdm(df_pos.groupby(['herb', 'Symptom']).size().items(),
                                   total=len(df_pos.groupby(['herb', 'Symptom']))):
        process_sample(herb, symptom, samples, herb_target_map, symptom_gene_map, ppi_dict, 1)

    print("processing negative samples...")
    for (herb, symptom), _ in tqdm(df_neg.groupby(['herb', 'Symptom']).size().items(),
                                   total=len(df_neg.groupby(['herb', 'Symptom']))):
        process_sample(herb, symptom, samples, herb_target_map, symptom_gene_map, ppi_dict, 0)

    return samples

def process_sample(herb, symptom, samples, herb_map, symptom_map, ppi_dict, label):
    """process single sample"""
    symbols = herb_map.get(herb, [])
    symptom_genes = symptom_map.get(symptom, [])

    herb_features = [np.mean([ppi_dict.get((s_gene, symbol), 0) for s_gene in symptom_genes]) or 0.0 for symbol in symbols]
    symptom_features = [np.mean([ppi_dict.get((s_gene, symbol), 0) for symbol in symbols]) or 0.0 for s_gene in symptom_genes]

    samples.append({
        'herb': herb,
        'symptom': symptom,
        'herb_symbols': symbols,
        'symptom_symbols': symptom_genes,
        'herb_features': herb_features,
        'symptom_features': symptom_features,
        'label': label
    })

def load_word2vec_model():
    return Word2Vec.load(DATA_PATHS['word2vec'])

def get_embedding(text, word2vec_model):
    try:
        return word2vec_model.wv[text]
    except KeyError:
        return np.zeros(128)