# Credit Card Fraud Detection Using Deep Autoencoder

## Project Overview

This project explores unsupervised anomaly detection for credit card fraud detection using a Deep Autoencoder.

The model is trained exclusively on normal transactions to learn their underlying patterns. Transactions with higher reconstruction errors are considered potential anomalies and may indicate fraudulent activity.

The project also investigates reconstruction-based feature attribution and quantitative explainability evaluation.

## Dataset

The project uses the Credit Card Fraud Detection dataset.

- Total transactions: 284,807
- Normal transactions: 284,315
- Fraudulent transactions: 492
- Fraud rate: 0.173%
- Input features: 31

The input features consist of:
- PCA-transformed features (V1–V28)
- Log-transformed transaction amount
- Cyclical time features (Sin_Time and Cos_Time)

## Methodology

The main workflow includes:

1. Dataset loading and temporal feature engineering
2. Chronological train, validation, and test splitting (70% / 15% / 15%)
3. Isolation of normal transactions for Autoencoder training
4. Robust feature scaling
5. Deep Autoencoder architecture design
6. Model training using reconstruction loss
7. Anomaly scoring and threshold calibration
8. Test-set evaluation
9. Reconstruction-based feature attribution
10. Quantitative XAI Fidelity evaluation

The Autoencoder is trained only on normal training transactions. Fraud detection is performed by analyzing reconstruction errors.

## Model Architecture

A fully connected Deep Autoencoder is implemented using PyTorch.

**Encoder:**
- Input dimension: 31
- Hidden layers: 24 and 16 neurons
- Latent dimension: 8

**Decoder:**
- Hidden layers: 16 and 24 neurons
- Output dimension: 31

The architecture uses Batch Normalization, LeakyReLU activation, and Dropout.

## Training Configuration

- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 512
- Maximum epochs: 30
- Loss function: Smooth L1 Loss (Huber Loss)
- Learning rate scheduler: ReduceLROnPlateau
- Training device: CUDA (GPU)

## Experimental Results

Two reconstruction-based scoring approaches were evaluated on the test set.

| Metric | Standard Reconstruction MSE | Feature-Normalized Top-K |
|---|---:|---:|
| ROC-AUC | 0.9199 | 0.9413 |
| PR-AUC | 0.0309 | 0.1783 |
| Precision | 5.24% | 33.33% |
| Recall | 44.23% | 40.38% |
| F1-Score | 0.0937 | 0.3652 |

The Feature-Normalized Top-K approach achieved better overall results in this experiment, particularly in PR-AUC, Precision, and F1-Score.

## Explainable AI (XAI)

The project includes reconstruction-based feature attribution to identify features contributing to anomalous reconstruction errors.

For one analyzed fraudulent transaction, the top contributing features were:

1. V17
2. V14
3. V16
4. V12
5. V3

A quantitative Fidelity evaluation was also performed on 21 detected fraud cases, reporting a score-drop-based Fidelity of 55.41%.

## Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Repository Structure

```text
credit-card-fraud-autoencoder/
│
├── Autoencoder.ipynb
└── README.md
```

## How to Run

1. Open the notebook in Google Colab.
2. Install the required libraries if necessary.
3. Run the notebook cells in order.
4. The dataset is downloaded automatically by the notebook.

## Notes

This project is an experimental implementation of unsupervised fraud detection using reconstruction-based anomaly scoring.

The reported results are specific to the dataset, chronological split, and evaluation procedures used in this notebook.
