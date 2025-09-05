import torch
from data_loader import load_data, generate_samples, load_word2vec_model
from model import initialize_model
from utils import HerbSymptomDataset, create_data_loaders
from train_eval import train_model, test_model, plot_metrics, save_metrics, load_metrics
from predict import run_prediction
from config import TRAIN_CONFIG, DEVICE, MODEL_SAVE_PATH, METRICS_SAVE_PATH

# Apply global seed at the very beginning
from train_eval import set_seed
set_seed()

def main():
    # 1. Load data
    print("Loading data...")
    df_pos, df_neg, df_herb, df_symptom, ppi_dict = load_data()
    
    # 2. Generate samples
    print("Generating samples...")
    all_samples = generate_samples(df_pos, df_neg, df_herb, df_symptom, ppi_dict)
    
    # 3. Load Word2Vec model
    word2vec_model = load_word2vec_model()
    
    # 4. Build dataset & loaders
    print("Creating dataset and data loaders...")
    dataset = HerbSymptomDataset(all_samples, word2vec_model)
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, TRAIN_CONFIG['batch_size']
    )
    
    # 5. Training flag
    TRAIN_MODE = True  # Set False to skip training and plot saved metrics
    
    if TRAIN_MODE:
        # 6. Initialize model
        model = initialize_model()
        
        # 7. Train & validate
        print("Starting training...")
        train_losses, train_accs, val_losses, val_aucs, val_accs = train_model(
            model, train_loader, val_loader, DEVICE, TRAIN_CONFIG['epochs']
        )
        
        # 8. Test model
        print("Testing model...")
        test_metrics, class_distribution = test_model(model, test_loader, DEVICE)
        
        # 9. Save metrics
        save_metrics(train_losses, train_accs, val_losses, val_aucs, val_accs,
                    test_metrics, class_distribution)
        
        # 10. Plot metrics
        plot_metrics(train_losses, val_losses, train_accs, val_accs,
                    test_metrics, class_distribution, 
                    save_path="initial_metrics.pdf")
    else:
        # Load saved metrics
        try:
            (train_losses, train_accs, 
             val_losses, val_aucs, val_accs,
             test_metrics, class_distribution) = load_metrics()
            
            plot_metrics(train_losses, val_losses, train_accs, val_accs,
                        test_metrics, class_distribution,
                        save_path="updated_style_metrics.pdf")
        except FileNotFoundError:
            print("Error: Metrics data not found. Please run in TRAIN_MODE first.")
    
    # 11. Run prediction
    print("Running prediction...")
    total_df = run_prediction(dataset)
    print("Prediction completed.")
    print(total_df.head())

if __name__ == "__main__":
    main()