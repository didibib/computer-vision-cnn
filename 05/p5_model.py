from tensorflow import keras
from tensorflow.keras import layers

p5 = keras.Sequential([
    layers.Conv2D(
        32, 3, 1, padding = 'same', activation = 'relu', input_shape = (112, 112, 3)),    
    layers.Conv2D(
        32, 3, 1, padding = 'valid', activation = 'relu'),    
    layers.MaxPool2D(
        (2,2), 2, padding = 'valid'),

    layers.Dropout(.1),

    layers.Conv2D(
        32, 3, 1, padding = 'same', activation = 'relu'),          
    layers.Conv2D(
        32, 3, 1, padding = 'valid', activation = 'relu'),  
    layers.MaxPool2D(
        (2,2), 2, padding = 'valid'),

    layers.Dropout(.1),
    
    layers.Conv2D(
        32, 3, 1, padding = 'same', activation = 'relu'),    
    layers.Conv2D(
        32, 3, 1, padding = 'valid', activation = 'relu'), 
    layers.MaxPool2D(
        (2,2), 2, padding = 'valid'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    # Output layer
    layers.Dense(40)
])