import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from model import create_model

def train_irrigation_model(dataset_path='irrigation_dataset.csv', 
                          model_output_path='irrigation_model.keras',
                          scaler_output_path='scaler.pkl',
                          epochs=50,
                          test_size=0.2,
                          random_state=42):
    """
    Train the ANN model on the irrigation dataset.
    
    Steps:
    ------
    1. Load the irrigation dataset
    2. Split features and target variable
    3. Split into training (80%) and testing (20%) sets
    4. Scale features using StandardScaler
    5. Create and train the model
    6. Evaluate on test data
    7. Save the trained model and scaler
    
    Parameters:
    -----------
    dataset_path : str
        Path to the irrigation dataset CSV file
    model_output_path : str
        Path to save the trained model
    scaler_output_path : str
        Path to save the fitted StandardScaler
    epochs : int
        Number of training epochs
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    """
    
    print("="*70)
    print("TRAINING HYBRID ANN FOR IRRIGATION OPTIMIZATION")
    print("="*70)
    
    print("\n[1/7] Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"\nFeatures: {list(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")
    
    print("\n[2/7] Splitting features and target...")
    X = df[['soil_moisture', 'temperature', 'humidity', 'crop_type']].values
    y = df['irrigation_duration_minutes'].values
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    print("\n[3/7] Splitting into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    print("\n[4/7] Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled successfully")
    print(f"Feature means: {scaler.mean_}")
    print(f"Feature std devs: {scaler.scale_}")
    
    print("\n[5/7] Creating the ANN model...")
    model = create_model(input_dim=X_train.shape[1])
    
    print("\n[6/7] Training the model...")
    print(f"Epochs: {epochs}")
    print(f"Optimizer: Adam")
    print(f"Loss Function: Mean Squared Error (MSE)")
    print("-"*70)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    print("\n[7/7] Evaluating model on test data...")
    test_loss, test_mse = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Mean Squared Error: {test_mse:.4f}")
    print(f"Test RMSE: {np.sqrt(test_mse):.4f} minutes")
    
    y_pred = model.predict(X_test_scaled, verbose=0)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    print(f"Test Mean Absolute Error: {mae:.4f} minutes")
    
    r2_score = 1 - (np.sum((y_test - y_pred.flatten())**2) / 
                    np.sum((y_test - np.mean(y_test))**2))
    print(f"R² Score: {r2_score:.4f}")
    
    print(f"\n[SAVING] Saving trained model to {model_output_path}...")
    model.save(model_output_path)
    print("Model saved successfully!")
    
    print(f"\n[SAVING] Saving scaler to {scaler_output_path}...")
    joblib.dump(scaler, scaler_output_path)
    print("Scaler saved successfully!")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel file: {model_output_path}")
    print(f"Scaler file: {scaler_output_path}")
    print("\nYou can now use predict.py to make predictions on new data.")
    
    return model, scaler, history

if __name__ == "__main__":
    train_irrigation_model()
