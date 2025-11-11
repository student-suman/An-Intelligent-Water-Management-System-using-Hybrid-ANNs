# Hybrid ANN for Agricultural Irrigation Optimization

## Overview
A machine learning project that implements a Hybrid Artificial Neural Network (ANN) to predict optimal irrigation duration for agricultural fields. The system uses a custom activation function combining Swish and Leaky ReLU properties to optimize water usage and reduce wastage.

## Project Purpose
To predict the required irrigation duration (in minutes) based on real-time environmental data including soil moisture, temperature, humidity, and crop type. This helps farmers avoid overwatering and underwatering, thus conserving water resources.

## Current State
**Status:** Fully implemented
**Date:** October 17, 2025

### Recent Changes (October 17, 2025)
- Created complete project structure with modular Python files
- Implemented custom Hybrid_Swish_ReLU activation function
- Built ANN model with Sequential architecture (Input → 64 neurons → 64 neurons → Output)
- Created synthetic dataset generator with 1000 samples
- Implemented training pipeline with data preprocessing and model evaluation
- Added prediction module for inference on new data

## Project Architecture

### File Structure
```
.
├── data_generator.py      # Synthetic dataset generation (1000 samples)
├── model.py               # Custom activation function & ANN architecture
├── train.py               # Training pipeline with data preprocessing
├── predict.py             # Inference module for new predictions
├── requirements.txt       # Python dependencies
├── irrigation_dataset.csv # Generated dataset (created at runtime)
├── irrigation_model.h5    # Trained model (created after training)
└── scaler.pkl            # Fitted StandardScaler (created after training)
```

### Key Components

#### 1. Dataset Generation (`data_generator.py`)
- Generates 1000 synthetic samples
- Features: soil_moisture (0-100%), temperature (10-45°C), humidity (20-90%), crop_type (0-2)
- Target: irrigation_duration_minutes (0-60 min)
- Formula: `60 - (soil_moisture * 0.5) + (temperature * 0.3) - (humidity * 0.2) + noise`

#### 2. Model Architecture (`model.py`)
- **Custom Activation:** Hybrid_Swish_ReLU
  - For x > 0: Swish behavior → f(x) = x * sigmoid(x)
  - For x ≤ 0: Leaky ReLU behavior → f(x) = 0.01 * x
- **ANN Structure:**
  - Input layer: 4 features
  - Hidden layer 1: 64 neurons + Hybrid_Swish_ReLU
  - Hidden layer 2: 64 neurons + Hybrid_Swish_ReLU
  - Output layer: 1 neuron + ReLU (ensures non-negative output)
- **Compilation:** Adam optimizer, MSE loss function

#### 3. Training Pipeline (`train.py`)
- 80-20 train-test split
- StandardScaler for feature normalization
- 50 epochs with batch size 32
- 10% validation split during training
- Saves trained model (.h5) and scaler (.pkl)
- Evaluation metrics: MSE, RMSE, MAE, R²

#### 4. Prediction System (`predict.py`)
- Loads trained model and scaler
- Accepts new environmental data
- Returns predicted irrigation duration
- Includes example predictions with formatted output

## Technical Stack
- **TensorFlow/Keras:** Deep learning framework
- **NumPy:** Numerical computations
- **Pandas:** Data manipulation
- **scikit-learn:** Data preprocessing (StandardScaler, train_test_split)
- **joblib:** Model persistence

## Usage Instructions

### 1. Generate Dataset
```bash
python data_generator.py
```

### 2. Train Model
```bash
python train.py
```

### 3. Make Predictions
```bash
python predict.py
```

Or programmatically:
```python
from predict import predict_irrigation

duration = predict_irrigation(
    soil_moisture=35,
    temperature=30,
    humidity=60,
    crop_type=1  # 0=Wheat, 1=Corn, 2=Rice
)
print(f"Predicted irrigation: {duration:.2f} minutes")
```

## Model Performance Expectations
- Expected MSE: < 10 (depending on noise level)
- Expected RMSE: < 3-4 minutes
- Expected R²: > 0.85

## Future Enhancements
- Cross-validation for improved robustness
- Learning curves and training history visualization
- Hyperparameter tuning (grid search/random search)
- Confidence intervals for predictions
- Extended evaluation metrics dashboard
- Input validation and error handling
- Real-world data integration

## Dependencies
See `requirements.txt` for full list:
- tensorflow
- pandas
- numpy
- scikit-learn
- joblib

## User Preferences
- Well-commented code for educational purposes
- Modular file structure for maintainability
- Comprehensive console output during training
- Example predictions for demonstration
