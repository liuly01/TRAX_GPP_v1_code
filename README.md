# LSTM-based GPP Prediction with STL Decomposition

This repository contains the implementation of a hybrid Long Short-Term Memory (LSTM) neural network model for predicting Gross Primary Productivity (GPP) using Seasonal-Trend decomposition using LOESS (STL). The model decomposes time series data into trend, seasonal, and residual components, and trains separate LSTM models for each component.

## Overview

This project implements a multi-component LSTM framework for monthly GPP prediction at EC-GPP sites. The key features include:

- **STL Decomposition**: Robust decomposition of GPP time series into trend, seasonal, and residual components
- **Component-wise Modeling**: Separate LSTM models for trend, seasonal, and residual components
- **Hyperparameter Optimization**: Automated hyperparameter tuning using Bayesian optimization (Hyperopt)
- **Cross-validation**: Time series cross-validation with expanding window strategy
- **Customized Loss Function**: Optional Inter-Annual Difference Constraint (IADC) for trend component to improve long-term predictions
- **Feature Attribution**: Integrated Gradients for model interpretability

## Repository Structure

```
├── main.py              # Main training and evaluation pipeline
├── models.py            # LSTM model architecture
├── data_utils.py        # Data preprocessing and dataset preparation
├── train_utils.py       # Training utilities and hyperparameter optimization
├── README.md            # This file
└── LICENSE              # License information
```

## Requirements

### Dependencies

- Python >= 3.8
- PyTorch >= 1.10.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.13.0
- hyperopt >= 0.2.7
- joblib >= 1.1.0

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install required packages:
```bash
pip install torch numpy pandas scikit-learn statsmodels hyperopt joblib
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

## Data Format

### Input Data

The model expects CSV files with the following structure:

- **Required columns**:
  - `time`: Timestamp for each observation
  - `GPP_DT_VUT_REF`: Target variable (Gross Primary Productivity)
  - Additional feature columns (meteorological variables, vegetation indices, etc.)

### Example

```csv
time,GPP_DT_VUT_REF,TA_F,SW_IN_F,VPD_F,P_F,...
2001-01-01,2.5,10.2,150.3,0.8,1.2,...
2001-02-01,3.1,12.5,180.7,1.1,0.5,...
...
```

## Usage

### 1. Configure Paths and Parameters

```python
BASE_DIR = r"/path/to/output/directory"
INPUT_DATA_DIR = r"/path/to/input/csv/files"

FEATURE_COLUMNS = [
    "TA_F",      # Air temperature
    "SW_IN_F",   # Shortwave radiation
    "VPD_F",     # Vapor pressure deficit
    # Add your feature columns here
]
```

### 2. Run Training

Execute the main script:

```bash
python main.py
```

The pipeline will automatically:
1. Load and preprocess data from all sites
2. Perform STL decomposition
3. Split data into development and test sets
4. Optimize hyperparameters using cross-validation
5. Train final models on the development set
6. Evaluate on the held-out test set
7. Save models, scalers, and predictions

### 3. Outputs

The following files will be generated in `BASE_DIR`:

**Models and Scalers**:
- `trend_model.pth`: Trained trend LSTM model
- `season_model.pth`: Trained seasonal LSTM model
- `resid_model.pth`: Trained residual LSTM model
- `*_feature_scaler.joblib`: Feature scalers for each component
- `*_target_scaler.joblib`: Target scalers for each component

**Predictions**:
- `{site_name}_test_results.csv`: Per-site test predictions
- `ALL_SITES_combined_test.csv`: Combined test results from all sites

**Feature Importance**:
- `trend_feature_importance.csv`: Feature attribution for trend model
- `season_feature_importance.csv`: Feature attribution for seasonal model
- `resid_feature_importance.csv`: Feature attribution for residual model

## Model Architecture

### LSTM Network

Each component (trend, seasonal, residual) uses a separate LSTM network:

```
Input (batch_size, window_length, num_features)
  ↓
LSTM (num_layers, hidden_size)
  ↓
Dropout (p)
  ↓
Linear (hidden_size → output_length)
  ↓
Output (batch_size, output_length)
```

### Hyperparameters

The following hyperparameters are optimized via Bayesian optimization:

- `hidden_size`: LSTM hidden dimension [32, 64, 128, 256]
- `num_layers`: Number of LSTM layers [1, 2, 3]
- `batch_size`: Training batch size [16, 32, 64, 128]
- `learning_rate`: Learning rate [1e-5, 1e-2]
- `weight_decay`: L2 regularization [1e-6, 1e-3]
- `p`: Dropout probability [0.0, 0.5]

## Key Parameters

Modify these parameters in `main.py` as needed:

```python
SEASONAL_PERIOD = 12      # Monthly data periodicity
INPUT_LENGTH = 12         # LSTM window length (months)
OUTPUT_LENGTH = 1         # Prediction horizon (1 month)
LAG_YOY = 12             # YoY lag for IADC
LAMBDA_YOY_DEFAULT = 1.0 # Weight for YoY loss (trend only)
HYP_EVALS = 50           # Hyperparameter search iterations
FINAL_EPOCHS = 300       # Maximum training epochs
```

## Advanced Features

### Year-over-Year (YoY) Constraint

For the trend component, you can enable Inter-Annual Difference Constraint (IADC) to improve long-term trend modeling:

```python
LAMBDA_YOY_DEFAULT = 1.0  
```

This adds an additional loss term that penalizes differences in predicted year-over-year changes.

### Custom Cross-Validation

The model uses an expanding window cross-validation strategy suitable for time series:

- Development set: First 80% of data (variable across sites)
- Test set: Last 20% of data (held out)
- CV folds: Expanding window with validation

## Known Issues

- GPU memory requirements scale with the number of sites and sequence length
- Large datasets may require adjusting batch size or sequence length

