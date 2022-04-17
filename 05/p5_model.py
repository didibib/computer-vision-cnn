from tensorflow import keras
from tensorflow.keras import layers

p5 = keras.Sequential([
    layers.Conv2D(
        32, 5, 1, padding = 'same', activation = 'relu', input_shape = (112, 112, 3)),    
    layers.MaxPool2D(
        (2,2), 2, padding = 'same'),

    layers.Conv2D(
        32, 5, 1, padding = 'same', activation = 'relu'),    
    layers.MaxPool2D(
        (2,2), 2, padding = 'same'),

    layers.Dropout(.3),
    layers.Conv2D(
        64, 3, 1, padding = 'same', activation = 'relu'),    
    layers.MaxPool2D(
        (2,2), 2, padding = 'same'),

    layers.Conv2D(
        64, 3, 1, padding = 'same', activation = 'relu'),
    layers.MaxPool2D(
        (2,2), 2, padding = 'same'),
    # layers.MaxPool2D(
    #     (2,2), 2, padding = 'same'),
    # layers.Conv2D(
    #     32, 3, 1, padding = 'same', activation = 'relu'),
    # layers.BatchNormalization(),
    # layers.MaxPool2D(
    #     (2,2), 2, padding = 'same'),
    # layers.Conv2D(
    #     32, 3, 1, padding = 'same', activation = 'relu'),
    # layers.BatchNormalization(),
    # layers.MaxPool2D(
    #     (2,2), 2, padding = 'same'),
        
    # layers.Conv2D(
    #     32, 3, 1, padding = 'same', activation = 'relu'),
    # layers.BatchNormalization(),
    # layers.MaxPool2D(
    #     (2,2), 2, padding = 'same'),
    
    layers.Flatten(),
    layers.Dense(80, activation='relu'),
    # Output layer
    layers.Dense(40)
])