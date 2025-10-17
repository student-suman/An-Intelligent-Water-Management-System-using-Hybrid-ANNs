import numpy as np
import joblib
from tensorflow import keras

def predict_irrigation(soil_moisture, temperature, humidity, crop_type,
                      model_path='irrigation_model.h5',
                      scaler_path='scaler.pkl'):
    """
    Predict the optimal irrigation duration for given environmental conditions.
    
    Parameters:
    -----------
    soil_moisture : float
        Soil moisture percentage (0-100%)
    temperature : float
        Temperature in Celsius (10-45°C)
    humidity : float
        Air humidity percentage (20-90%)
    crop_type : int
        Crop type (0=Wheat, 1=Corn, 2=Rice)
    model_path : str
        Path to the trained model file
    scaler_path : str
        Path to the fitted scaler file
    
    Returns:
    --------
    float : Predicted irrigation duration in minutes
    """
    
    model = keras.models.load_model(
        model_path,
        custom_objects={'HybridSwishReLU': keras.layers.Layer}
    )
    
    scaler = joblib.load(scaler_path)
    
    input_data = np.array([[soil_moisture, temperature, humidity, crop_type]])
    
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled, verbose=0)
    
    predicted_duration = prediction[0][0]
    
    return predicted_duration

def display_prediction(soil_moisture, temperature, humidity, crop_type):
    """
    Display a formatted prediction with input parameters.
    
    Parameters:
    -----------
    soil_moisture : float
        Soil moisture percentage
    temperature : float
        Temperature in Celsius
    humidity : float
        Air humidity percentage
    crop_type : int
        Crop type (0=Wheat, 1=Corn, 2=Rice)
    """
    
    crop_names = {0: 'Wheat', 1: 'Corn', 2: 'Rice'}
    
    print("\n" + "="*70)
    print("IRRIGATION PREDICTION")
    print("="*70)
    print("\nInput Parameters:")
    print(f"  - Soil Moisture: {soil_moisture}%")
    print(f"  - Temperature: {temperature}°C")
    print(f"  - Humidity: {humidity}%")
    print(f"  - Crop Type: {crop_names.get(crop_type, 'Unknown')} (code: {crop_type})")
    
    try:
        duration = predict_irrigation(soil_moisture, temperature, humidity, crop_type)
        
        print("\n" + "-"*70)
        print(f"PREDICTED IRRIGATION DURATION: {duration:.2f} minutes")
        print("-"*70)
        
        if duration < 15:
            recommendation = "Low irrigation needed - soil has good moisture"
        elif duration < 30:
            recommendation = "Moderate irrigation recommended"
        elif duration < 45:
            recommendation = "High irrigation needed - soil is relatively dry"
        else:
            recommendation = "Very high irrigation needed - critical soil dryness"
        
        print(f"\nRecommendation: {recommendation}")
        print("="*70 + "\n")
        
        return duration
        
    except FileNotFoundError as e:
        print(f"\nERROR: Model or scaler file not found!")
        print("Please run train.py first to train the model.")
        print(f"Details: {e}")
        return None
    except Exception as e:
        print(f"\nERROR: Prediction failed!")
        print(f"Details: {e}")
        return None

if __name__ == "__main__":
    print("="*70)
    print("HYBRID ANN IRRIGATION PREDICTION SYSTEM")
    print("="*70)
    
    print("\n[Example 1] Moderate conditions")
    display_prediction(
        soil_moisture=35,
        temperature=30,
        humidity=60,
        crop_type=1
    )
    
    print("\n[Example 2] High moisture, low temperature")
    display_prediction(
        soil_moisture=80,
        temperature=15,
        humidity=70,
        crop_type=0
    )
    
    print("\n[Example 3] Low moisture, high temperature")
    display_prediction(
        soil_moisture=20,
        temperature=40,
        humidity=30,
        crop_type=2
    )
    
    print("\n[Example 4] Optimal conditions")
    display_prediction(
        soil_moisture=60,
        temperature=25,
        humidity=55,
        crop_type=1
    )
