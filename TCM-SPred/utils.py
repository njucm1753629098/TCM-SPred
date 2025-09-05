import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import TRAIN_CONFIG, DEVICE, MODEL_SAVE_PATH, METRICS_SAVE_PATH
from data_loader import get_embedding

# Global seed for reproducibility
SEED = 42

class HerbSymptomDataset(Dataset):
    def __init__(self, samples, word2vec_model):
        self.samples = samples
        self.word2vec_model = word2vec_model
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Main embeddings
        herb_embed = get_embedding(sample['herb'], self.word2vec_model)
        symptom_embed = get_embedding(sample['symptom'], self.word2vec_model)
        
        # Herb symbol features
        herb_symbol_features = [
            np.concatenate([get_embedding(s, self.word2vec_model), [p]])
            for s, p in zip(sample['herb_symbols'], sample['herb_features'])
        ] if sample['herb_symbols'] else []
        
        # Symptom symbol features
        symptom_symbol_features = [
            np.concatenate([get_embedding(s, self.word2vec_model), [p]])
            for s, p in zip(sample['symptom_symbols'], sample['symptom_features'])
        ] if sample['symptom_symbols'] else []
        
        return {
            'herb': herb_embed.astype(np.float32),
            'herb_name': sample['herb'],
            'symptom': symptom_embed.astype(np.float32),
            'herb_symbols': np.array(herb_symbol_features).astype(np.float32) if herb_symbol_features else np.zeros((0, 129), dtype=np.float32),
            'symptom_symbols': np.array(symptom_symbol_features).astype(np.float32) if symptom_symbol_features else np.zeros((0, 129), dtype=np.float32),
            'label': np.array([sample['label']]).astype(np.float32)
        }

def dynamic_collate_fn(batch):
    """Dynamic batching with deterministic padding"""
    herb = torch.from_numpy(np.stack([b['herb'] for b in batch]))
    symptom = torch.from_numpy(np.stack([b['symptom'] for b in batch]))
    labels = torch.from_numpy(np.concatenate([b['label'] for b in batch])).unsqueeze(1)
    herb_names = [b['herb_name'] for b in batch]
    
    def pad(key):
        feats = [torch.from_numpy(b[key]) for b in batch]
        lengths = torch.tensor([len(f) for f in feats], dtype=torch.long)
        padded = pad_sequence(feats, batch_first=True, padding_value=0)
        return padded, lengths
    
    herb_symbols, herb_lengths = pad('herb_symbols')
    symptom_symbols, symptom_lengths = pad('symptom_symbols')
    
    return {
        'herb': herb, 'symptom': symptom,
        'herb_symbols': herb_symbols, 'herb_lengths': herb_lengths,
        'symptom_symbols': symptom_symbols, 'symptom_lengths': symptom_lengths,
        'labels': labels, 'herb_name': herb_names
    }

def create_data_loaders(dataset, batch_size):
    """Create data loaders with fixed random seed for reproducibility"""
    # Calculate dataset splits
    total_size = len(dataset)
    train_size = int(TRAIN_CONFIG['train_split'] * total_size)
    val_size = int(TRAIN_CONFIG['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    # Create generator with fixed seed
    generator = torch.Generator().manual_seed(SEED)
    
    # Split dataset with fixed seed
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create deterministic data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=dynamic_collate_fn,
        generator=generator
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        collate_fn=dynamic_collate_fn
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        collate_fn=dynamic_collate_fn
    )
    
    print(f"Data loaders ready: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    return train_loader, val_loader, test_loader

def save_metrics(train_losses, train_accs, val_losses, val_aucs, val_accs,
                 test_metrics, class_distribution, filename=METRICS_SAVE_PATH):
    """Save training metrics to file"""
    torch.save({
        'train_losses': train_losses, 'train_accs': train_accs,
        'val_losses': val_losses, 'val_aucs': val_aucs, 'val_accs': val_accs,
        'test_metrics': test_metrics, 'class_distribution': class_distribution
    }, filename)
    print(f"Metrics saved to {filename}")

def load_metrics(filename=METRICS_SAVE_PATH):
    """Load saved metrics from file"""
    data = torch.load(filename)
    return (data['train_losses'], data['train_accs'],
            data['val_losses'], data['val_aucs'], data['val_accs'],
            data['test_metrics'], data['class_distribution'])