import tensorflow as tf

# ===== VARIANT 1 =====
# The first thing we wanted to try was to introduce another dense layer at the end of our network.
# This will allow our network to make more non-linear decisions
variant1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'same',
        activation = 'relu',
        input_shape = (28, 28, 1) 
    ),

    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'valid',
        activation = 'relu',
    ),

    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid',
    ),
    
    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'same'
    ),

    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        activation = 'relu',
        padding = 'valid'
    ),

    tf.keras.layers.Dropout(.25),
    
    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid'
    ),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# ===== VARIANT 2 =====
# Smaller kernel size for our convolutional layers
# https://medium.com/analytics-vidhya/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-86a55a1e2d15
variant2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32,
        3,
        1,
        padding = 'same',
        input_shape = (28, 28, 1),
        activation = 'relu'
    ),

    tf.keras.layers.Conv2D(
        32,
        3,
        1,
        padding = 'valid',
        activation = 'relu'
    ),

    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid',
    ),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Conv2D(
        32,
        3,
        1,
        padding = 'same',
        activation = 'relu'
    ),

    tf.keras.layers.Conv2D(
        32,
        3,
        1,
        activation = 'relu',
        padding = 'valid'
    ),

    tf.keras.layers.Dropout(.25),
    
    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid'
    ),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# ===== VARIANT 3 =====
# The ReLU activation function maps all negative values to 0, creating dead neurons. If this happens often then a part 
# of the network will not contribute and basically will not be used. Leaky ReLU solves this issue by having a slope in the
# negative half axis. This improves the convergence of the network
# https://dl.acm.org/doi/10.1145/3433996.3434001
variant3 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'same',
        activation = tf.keras.layers.LeakyReLU(alpha=0.01),
        input_shape = (28, 28, 1) 
    ),

    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'valid',
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    ),

    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid'
    ),
    
    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'same',
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    ),

    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        activation = tf.keras.layers.LeakyReLU(alpha=0.01),
        padding = 'valid'
    ),

    tf.keras.layers.Dropout(.25),
    
    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid'
    ),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(10)
])

# ===== VARIANT 4 =====
# Some of the features coming out of a Conv2D might be negative and might be truncated by a non-linearity like ReLU
# If we normalize before activation we are including these negative values in the normalization immediately, 
# before culling them from the feature space.
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
variant4 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'same',
        input_shape = (28, 28, 1) 
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'valid',
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid',
    ),
    
    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'same',
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'valid'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Dropout(.25),
    
    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid'
    ),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])