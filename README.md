# TCM-SPred
A multimodal AI-driven Traditional Chinese Medicine (TCM) symptom prediction model (MTC-SPred) .

## Hardware & Software Environment:
- CPU: 14 vCPU Intel® Xeon® Gold 6330 @ 2.00 GHz  
- GPU: NVIDIA RTX 3090 (24 GB)  
- OS: Ubuntu 22.04 LTS  
- Python: 3.10  
- PyTorch: 2.1.2  
- CUDA: 11.8  

## Folder Structure
TCM-SPred/
├── data/                          # Training data
├── data_process/                  # Data preprocessing scripts
├── predictions/                   # Prediction results and figures
├── training_metrics_plots/        # Additional evaluation metric plots
├── word2vec_models/               # Pre-trained word embedding models
└── *.py                           # Main scripts

## ## Usage
### 1. Train the TCM-SPred model：
```bash
python train.py
```
### 2. 10-fold cross-validation & plots(delete `fold_results.pkl` if you want to re-train from scratch)
```bash
python cross_validation.py
```
### 3. Additional evaluation figures
```bash
python plot_metrics_draw.py
```
### 4. Renal-fibrosis-related symptom prediction
```bash
python predict_rf_symptoms.py
```
### 5. Top-10 herb rank distribution for renal fibrosis
```bash
python plot2_draw.py
```

## Contact
For questions about this project, please open an issue on GitHub or contact [20231066@njucm.edu.cn].

