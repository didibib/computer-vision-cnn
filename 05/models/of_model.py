from tensorflow import keras
from tensorflow.keras import layers

of = keras.Sequential([
    layers.Conv3D(
        8, (5,5,3), (2,2,1), padding = 'valid', activation = 'relu', input_shape = (164, 164, 16, 2)),    
    layers.Conv3D(
        16, (3,3,3), 1, padding = 'valid', activation = 'relu'),    
    layers.MaxPool3D(
        (2,2,2), 2, padding = 'valid'),

    layers.Conv3D(
        32, (3,3,3), (2,2,1), padding = 'same', activation = 'relu'),          
    layers.Conv3D(
        64, (3,3,3), 1, padding = 'valid', activation = 'relu'),  
    layers.MaxPool3D(
        (2,2,4), 2, padding = 'valid'),
    layers.BatchNormalization(), 
    
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer = keras.regularizers.l2(0.001), kernel_initializer= 'he_uniform'),
    
    layers.Dropout(.5),
    layers.Dense(4, activation='softmax')
])