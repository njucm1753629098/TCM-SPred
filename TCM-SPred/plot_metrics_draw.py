from train_eval import plot_from_saved_metrics
from config import METRICS_SAVE_PATH

def main():
    print("Plotting all metrics from saved data...")
    plot_from_saved_metrics(
        metrics_path=METRICS_SAVE_PATH,
        save_folder="training_metrics_plots"
    )
    print("All metrics plots saved to 'training_metrics_plots' folder")

if __name__ == "__main__":
    main()