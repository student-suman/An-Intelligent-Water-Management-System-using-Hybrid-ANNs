import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from model import create_model
import matplotlib.pyplot as plt

# CHANGED: The function now accepts paths for the new model and scaler
def train_irrigation_model(dataset_path='irrigation_dataset.csv', 
                           model_output_path='irrigation_model.keras',
                           scaler_output_path='scaler.pkl',
                           epochs=50,
                           test_size=0.2,
                           random_state=42):
    """
    Train the ANN model on the irrigation dataset and return performance metrics.
    """
    
    # ... (The rest of the loading and preprocessing code is the same)
    df = pd.read_csv(dataset_path)
    X = df[['soil_moisture', 'temperature', 'humidity', 'crop_type']].values
    y = df['irrigation_duration_minutes'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = create_model(input_dim=X_train.shape[1])
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0 # Set verbose to 0 to keep the app UI clean
    )
    
    # Evaluation
    test_loss, test_mse = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_pred = model.predict(X_test_scaled, verbose=0)
    
    rmse = np.sqrt(test_mse)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    r2_score = 1 - (np.sum((y_test - y_pred.flatten())**2) / 
                    np.sum((y_test - np.mean(y_test))**2))
    
    # Save the new model and scaler to the specified paths
    model.save(model_output_path)
    joblib.dump(scaler, scaler_output_path)
    
    # Return performance metrics and history for display in the app
    performance = {'R² Score': r2_score, 'RMSE (min)': rmse, 'MAE (min)': mae}
    
    return performance, history

if __name__ == "__main__":
    # This part is for running the script directly, not used by the Streamlit app
    performance, history = train_irrigation_model()
    print("Training Complete!")
    print("Performance Metrics:", performance)
    
    # Plotting logic for direct execution
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_history.png")
    print("Plot saved as training_history.png")