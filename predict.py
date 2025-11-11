import numpy as np
import joblib
from tensorflow import keras
from model import HybridSwishReLU

def predict_irrigation(soil_moisture, temperature, humidity, crop_type,
                      model_path='irrigation_model.keras',
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
        custom_objects={'HybridSwishReLU': HybridSwishReLU}
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

# import argparse # <-- NEW: For creating a command-line interface
# import sys      # <-- NEW: For exiting the script on bad input
# import numpy as np
# import joblib
# from tensorflow import keras
# from model import HybridSwishReLU

# # --- Constants ---
# MODEL_PATH = 'irrigation_model.keras' # <-- NEW: Define constants for file paths
# SCALER_PATH = 'scaler.pkl'

# # --- Core Prediction Function ---
# # Now accepts the loaded model and scaler as arguments for efficiency
# def predict_irrigation(
#     soil_moisture: float,
#     temperature: float,
#     humidity: float,
#     crop_type: int,
#     model: keras.Model, # <-- CHANGED: Accept loaded model
#     scaler: object      # <-- CHANGED: Accept loaded scaler
# ) -> float:
#     """Predict the optimal irrigation duration."""
    
#     input_data = np.array([[soil_moisture, temperature, humidity, crop_type]])
#     input_scaled = scaler.transform(input_data)
    
#     prediction = model.predict(input_scaled, verbose=0)
#     predicted_duration = prediction[0][0]
    
#     return predicted_duration

# # --- Display Function ---
# def display_prediction(
#     soil_moisture: float,
#     temperature: float,
#     humidity: float,
#     crop_type: int,
#     model: keras.Model,
#     scaler: object
# ):
#     """Display a formatted prediction with input parameters."""
    
#     crop_names = {0: 'Wheat', 1: 'Corn', 2: 'Rice'}
    
#     print("\n" + "="*70)
#     print("IRRIGATION PREDICTION")
#     print("="*70)
#     print("\nInput Parameters:")
#     print(f"  - Soil Moisture: {soil_moisture}%")
#     print(f"  - Temperature: {temperature}°C")
#     print(f"  - Humidity: {humidity}%")
#     print(f"  - Crop Type: {crop_names.get(crop_type, 'Unknown')} (code: {crop_type})")
    
#     # <-- CHANGED: Pass the loaded model/scaler to the function
#     duration = predict_irrigation(soil_moisture, temperature, humidity, crop_type, model, scaler)
    
#     print("\n" + "-"*70)
#     print(f"PREDICTED IRRIGATION DURATION: {duration:.2f} minutes")
#     print("-"*70)
    
#     if duration < 15:
#         recommendation = "Low irrigation needed - soil has good moisture"
#     elif duration < 30:
#         recommendation = "Moderate irrigation recommended"
#     elif duration < 45:
#         recommendation = "High irrigation needed - soil is relatively dry"
#     else:
#         recommendation = "Very high irrigation needed - critical soil dryness"
    
#     print(f"\nRecommendation: {recommendation}")
#     print("="*70 + "\n")

# # --- Main Execution Block ---
# def main(): # <-- NEW: Main function to orchestrate everything
#     """
#     Main function to handle argument parsing, model loading, and prediction.
#     """
#     # 1. Set up the Command-Line Interface (CLI)
#     parser = argparse.ArgumentParser(
#         description="Predict optimal irrigation duration using a Hybrid ANN model.",
#         formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
#     )
#     parser.add_argument("soil_moisture", type=float, help="Soil moisture percentage (0-100)")
#     parser.add_argument("temperature", type=float, help="Ambient temperature in Celsius (10-45)")
#     parser.add_argument("humidity", type=float, help="Air humidity percentage (20-90)")
#     parser.add_argument("crop_type", type=int, choices=[0, 1, 2], help="Crop type (0: Wheat, 1: Corn, 2: Rice)")
    
#     args = parser.parse_args()

#     # 2. Validate User Inputs
#     if not (0 <= args.soil_moisture <= 100):
#         print("Error: Soil moisture must be between 0 and 100.", file=sys.stderr)
#         sys.exit(1)
#     if not (10 <= args.temperature <= 45):
#         print("Error: Temperature must be between 10 and 45.", file=sys.stderr)
#         sys.exit(1)
#     if not (20 <= args.humidity <= 90):
#         print("Error: Humidity must be between 20 and 90.", file=sys.stderr)
#         sys.exit(1)

#     # 3. Load Model and Scaler (only once)
#     try:
#         model = keras.models.load_model(
#             MODEL_PATH,
#             custom_objects={'HybridSwishReLU': HybridSwishReLU}
#         )
#         scaler = joblib.load(SCALER_PATH)
#     except FileNotFoundError:
#         print(f"Error: Model or scaler file not found! Please run train.py first.", file=sys.stderr)
#         sys.exit(1)
#     except Exception as e:
#         print(f"Error loading model or scaler: {e}", file=sys.stderr)
#         sys.exit(1)

#     # 4. Run and Display Prediction
#     display_prediction(
#         args.soil_moisture,
#         args.temperature,
#         args.humidity,
#         args.crop_type,
#         model, # Pass the loaded objects
#         scaler
#     )

# if __name__ == "__main__":
#     main()
