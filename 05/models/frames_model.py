from tensorflow import keras
from tensorflow.keras import layers

framesv2 = keras.Sequential([
    layers.Conv2D(
        16, 9, 4, activation = 'relu', input_shape = (164, 164, 3)),    
    layers.MaxPool2D(3, 2),

    layers.Conv2D(
        64, 5, 1, padding = 'same', activation = 'relu'), 
    layers.MaxPool2D(3, 2),
    layers.BatchNormalization(),

    layers.Conv2D(
        96, 3, 1, padding = 'same', activation = 'relu'), 
    layers.Conv2D(
        96, 3, 1, padding = 'same', activation = 'relu'), 
    layers.Conv2D(
        64, 3, 1, padding = 'same', activation = 'relu'), 
    layers.MaxPool2D(3, 2),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dropout(.5),
    layers.Dense(128, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001), kernel_initializer= 'he_uniform'),
    layers.Dropout(.5),
    layers.Dense(128, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001)),
    layers.Dense(40, activation='softmax')
])



frames = keras.Sequential([
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