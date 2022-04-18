from tensorflow import keras
from tensorflow.keras import layers

p5 = keras.Sequential([
    layers.Conv2D(
        16, 3, 1, padding = 'same', activation = 'relu', input_shape = (164, 164, 3)),    
    layers.Conv2D(
        16, 3, 1, padding = 'valid', activation = 'relu'),    
    layers.BatchNormalization(),
    layers.MaxPool2D(
        (2,2), 2, padding = 'valid'),

    layers.Dropout(.3),

    layers.Conv2D(
        16, 3, 1, padding = 'same', activation = 'relu'),          
    layers.Conv2D(
        16, 3, 1, padding = 'valid', activation = 'relu'),  
    layers.BatchNormalization(),
    layers.MaxPool2D(
        (2,2), 2, padding = 'valid'),

    layers.Dropout(.3),

    layers.Conv2D(
        16, 3, 1, padding = 'same', activation = 'relu'),    
    layers.Conv2D(
        16, 3, (2,2), padding = 'valid', activation = 'relu'), 
    layers.BatchNormalization(),
    layers.MaxPool2D(
        (2,2), 2, padding = 'valid'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001), kernel_initializer= 'he_uniform'),
    layers.Dense(128, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)),
    # Output layer
    layers.Dense(40, activation='softmax')
])