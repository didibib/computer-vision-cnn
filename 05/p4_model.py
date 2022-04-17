from tensorflow import keras
from tensorflow.keras import layers

p4 = keras.Sequential([
    # We don't use an input layer since we specify the input_shape in the next layer (conv2D in this case)
    # Our input is of size 28x28x1, our output will be 24x24x32, since we have no padding and use 32 filters
    layers.Conv2D(
        32,                           # Amount of filters this layer uses. Creates a depth of x
        5,                            # Size of the nxn filter, use one value to use it for all spatial dimensions
        1,                            # Stride
        padding = 'same',             # Use no padding
        activation = 'relu',
        input_shape = (28, 28, 1) # Batch size of 32, Images are 28x28, 1 channel (greyscale)
    ),

    # Another convolution layer. This creates more flexibility in expressing non-linear transformations without losing information,
    # as found here: https://stackoverflow.com/questions/46515248/intuition-behind-stacking-multiple-conv2d-layers-before-dropout-in-cnn
    layers.Conv2D(
        32,                           # Amount of filters this layer uses. Creates a depth of x
        5,                            # Size of the nxn filter, use one value to use it for all spatial dimensions
        1,                            # Stride
        padding = 'valid',            # Use no padding
        activation = 'relu'
    ),

    # Pooling layer to achieve some translation invariance, as well as reduce resolution
    # Input will be of size 26x26x5, output will be 13x13x32
    layers.MaxPool2D(
        (2,2),            # Kernel size
        2,                # Stride
        padding = 'valid', # No padding
    ),
    
    # Dropout layer which will drop 25% of the inputs during training
    # Does not alter the input shape
    # Add randomization to prevent overfitting 
    layers.Dropout(.25),

    # Extract higher level features
    layers.Conv2D(
        32,
        5,
        1,
        padding = 'same',
        activation = 'relu',
    ),

    # Again, we stack our convolation layers to get more non-linear flexibility
    layers.Conv2D(
        32,
        5,
        1,
        padding = 'valid',
        activation = 'relu', # Use relu activation here
    ),

    # Introduce more randomisation to prevent overfitting
    layers.Dropout(.25),
    
    #  Will change shape from 13x13x32 to 6x6x32
    layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid',
    ),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # Output layer
    layers.Dense(10)
])