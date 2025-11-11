import serial
import time
import numpy as np
import joblib
from tensorflow import keras
from model import HybridSwishReLU

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'  # IMPORTANT: Change this to your Arduino's port!
BAUD_RATE = 9600
MODEL_PATH = 'irrigation_model.keras'
SCALER_PATH = 'scaler.pkl'

# --- Load the AI Model and Scaler ---
try:
    model = keras.models.load_model(MODEL_PATH, custom_objects={'HybridSwishReLU': HybridSwishReLU})
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] AI model and scaler loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load model or scaler. {e}")
    exit()

def get_prediction(soil_moisture, temperature, humidity, crop_type=1):
    """Feeds sensor data to the AI model and returns the duration."""
    input_data = np.array([[soil_moisture, temperature, humidity, crop_type]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)
    return prediction[0][0]

# --- Main Loop to Run the System ---
try:
    # Connect to the Arduino
    arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=2)
    print(f"[INFO] Connected to Arduino on port {SERIAL_PORT}.")
    time.sleep(2) # Wait for the connection to establish

    while True:
        # 1. Read sensor data from Arduino
        line = arduino.readline().decode('utf-8').strip()
        
        if line:
            print(f"Received from Arduino: {line}")
            try:
                # 2. Parse the data
                soil, temp, hum = map(float, line.split(','))
                
                # 3. Get a prediction from the AI model
                duration_minutes = get_prediction(soil, temp, hum)
                duration_seconds = duration_minutes * 60

                print(f"[AI PREDICTION] Recommended duration: {duration_minutes:.2f} minutes.")
                
                # 4. Make a decision
                # Let's decide to water if the recommended time is more than 5 minutes
                if duration_minutes > 5:
                    print(f"[ACTION] Watering for {duration_seconds:.1f} seconds...")
                    
                    # Send 'ON' command to Arduino
                    arduino.write(b'ON\n')
                    
                    # Wait for the predicted duration
                    time.sleep(duration_seconds)
                    
                    # Send 'OFF' command to Arduino
                    arduino.write(b'OFF\n')
                    print("[ACTION] Watering complete.")
                else:
                    print("[ACTION] Soil is moist enough. No watering needed.")

            except (ValueError, IndexError):
                print(f"[WARNING] Could not parse data: '{line}'. Skipping.")
        
        # Wait before the next cycle
        print("-" * 30)
        time.sleep(10) # Check every 10 seconds

except serial.SerialException as e:
    print(f"[ERROR] Could not connect to Arduino on {SERIAL_PORT}. Please check the port and connection.")
    print(f"Details: {e}")
except KeyboardInterrupt:
    print("\n[INFO] Program terminated by user. Closing connection.")
finally:
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("[INFO] Arduino connection closed.")