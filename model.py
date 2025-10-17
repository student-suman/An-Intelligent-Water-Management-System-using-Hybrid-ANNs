import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def hybrid_swish_relu(x):
    """
    Custom Hybrid Activation Function: Hybrid_Swish_ReLU
    
    This function combines the properties of Swish and Leaky ReLU:
    - For x > 0: Behaves like Swish: f(x) = x * sigmoid(x)
    - For x <= 0: Behaves like Leaky ReLU: f(x) = 0.01 * x
    
    This hybrid approach allows:
    1. Smooth, non-monotonic behavior for positive values (Swish)
    2. Prevents dying neurons for negative values (Leaky ReLU)
    
    Parameters:
    -----------
    x : tensor
        Input tensor
    
    Returns:
    --------
    tensor : Activated output
    """
    swish_part = x * tf.nn.sigmoid(x)
    
    leaky_relu_part = 0.01 * x
    
    condition = tf.greater(x, 0)
    
    return tf.where(condition, swish_part, leaky_relu_part)

class HybridSwishReLU(layers.Layer):
    """
    Keras Layer wrapper for the Hybrid_Swish_ReLU activation function.
    This allows the custom activation to be used seamlessly in Keras models.
    """
    def __init__(self, **kwargs):
        super(HybridSwishReLU, self).__init__(**kwargs)
    
    def call(self, inputs):
        return hybrid_swish_relu(inputs)
    
    def get_config(self):
        return super(HybridSwishReLU, self).get_config()

def create_model(input_dim=4):
    """
    Create and compile the ANN model for irrigation duration prediction.
    
    Architecture:
    -------------
    - Input Layer: Accepts 4 features (soil_moisture, temperature, humidity, crop_type)
    - Hidden Layer 1: 64 neurons with Hybrid_Swish_ReLU activation
    - Hidden Layer 2: 64 neurons with Hybrid_Swish_ReLU activation
    - Output Layer: 1 neuron with ReLU activation (ensures non-negative predictions)
    
    Compilation:
    ------------
    - Optimizer: Adam
    - Loss Function: Mean Squared Error (MSE)
    - Metrics: MSE for monitoring
    
    Parameters:
    -----------
    input_dim : int
        Number of input features (default: 4)
    
    Returns:
    --------
    model : keras.Model
        Compiled Keras model ready for training
    """
    
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(64),
        HybridSwishReLU(),
        
        layers.Dense(64),
        HybridSwishReLU(),
        
        layers.Dense(1, activation='relu')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mse']
    )
    
    print("Model Architecture:")
    model.summary()
    
    return model

if __name__ == "__main__":
    print("Testing Hybrid_Swish_ReLU activation function...")
    test_input = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
    output = hybrid_swish_relu(test_input)
    print(f"Input: {test_input.numpy()}")
    print(f"Output: {output.numpy()}")
    
    print("\nCreating model...")
    model = create_model()
