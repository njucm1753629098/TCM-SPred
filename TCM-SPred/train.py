import torch
from data_loader import load_data, generate_samples, load_word2vec_model
from model import initialize_model
from utils import HerbSymptomDataset, create_data_loaders, save_metrics
from train_eval import train_model, test_model
from config import TRAIN_CONFIG, DEVICE, MODEL_SAVE_PATH, METRICS_SAVE_PATH


def main():
    # 1. load data
    print("Loading data...")
    df_pos, df_neg, df_herb, df_symptom, ppi_dict = load_data()

    # 2. generate samples
    print("Generating samples...")
    all_samples = generate_samples(df_pos, df_neg, df_herb, df_symptom, ppi_dict)

    # 3. load Word2Vec
    word2vec_model = load_word2vec_model()

    # 4. build dataset & loaders
    print("Creating dataset and data loaders...")
    dataset = HerbSymptomDataset(all_samples, word2vec_model)
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, TRAIN_CONFIG['batch_size']
    )

    # 5. init model
    model = initialize_model()

    # 6. train & validate
    print("Starting training...")
    train_losses, train_accs, val_losses, val_aucs, val_accs = train_model(
        model, train_loader, val_loader, DEVICE, TRAIN_CONFIG['epochs']
    )

    # 7. test
    print("Testing model...")
    test_metrics, class_distribution = test_model(model, test_loader, DEVICE)

    # 8. save metrics
    save_metrics(train_losses, train_accs, val_losses, val_aucs, val_accs,
                 test_metrics, class_distribution)
    print(f"Training completed. Metrics saved to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()