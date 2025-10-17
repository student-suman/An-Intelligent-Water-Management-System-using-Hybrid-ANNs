import numpy as np
import pandas as pd

def generate_irrigation_dataset(num_samples=1000, output_file='irrigation_dataset.csv'):
    """
    Generate a synthetic irrigation dataset with environmental features
    and calculated irrigation duration targets.
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate (default: 1000)
    output_file : str
        Output CSV filename (default: 'irrigation_dataset.csv')
    
    Features:
    ---------
    - soil_moisture: Soil moisture percentage (0-100%)
    - temperature: Temperature in Celsius (10-45°C)
    - humidity: Air humidity percentage (20-90%)
    - crop_type: Categorical crop type (0=Wheat, 1=Corn, 2=Rice)
    
    Target:
    -------
    - irrigation_duration_minutes: Required irrigation time (0-60 minutes)
    """
    
    np.random.seed(42)
    
    soil_moisture = np.random.uniform(0, 100, num_samples)
    temperature = np.random.uniform(10, 45, num_samples)
    humidity = np.random.uniform(20, 90, num_samples)
    crop_type = np.random.randint(0, 3, num_samples)
    
    irrigation_duration = (
        60 - (soil_moisture * 0.5) + 
        (temperature * 0.3) - 
        (humidity * 0.2)
    )
    
    noise = np.random.normal(0, 3, num_samples)
    irrigation_duration = irrigation_duration + noise
    
    irrigation_duration = np.clip(irrigation_duration, 0, 60)
    
    dataset = pd.DataFrame({
        'soil_moisture': soil_moisture,
        'temperature': temperature,
        'humidity': humidity,
        'crop_type': crop_type,
        'irrigation_duration_minutes': irrigation_duration
    })
    
    dataset.to_csv(output_file, index=False)
    print(f"Dataset generated successfully with {num_samples} samples!")
    print(f"Saved to: {output_file}")
    print("\nDataset preview:")
    print(dataset.head())
    print("\nDataset statistics:")
    print(dataset.describe())
    
    return dataset

if __name__ == "__main__":
    generate_irrigation_dataset()
