import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

import util
import callback

from keras.datasets import fashion_mnist

print("-- Starting program")

# Import training and validation sets
print("-- Import fashion mnist data set")
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("-- Normalize colors")
train_images = train_images / 255.0
test_images = test_images / 255.0

print("-- Creating base model")
base_model = tf.keras.Sequential([
    # We don't use an input layer since we specify the input_shape in the next layer (conv2D in this case)
    # Our input is of size 28x28x1, our output will be 24x24x5, since we have no padding and use 5 filters
    tf.keras.layers.Conv2D(
        32,                           # Amount of filters this layer uses. Creates a depth of x
        5,                            # Size of the nxn filter, use one value to use it for all spatial dimensions
        1,                            # Stride
        padding = 'same',             # Use no padding
        input_shape = (28, 28, 1) # Batch size of 32, Images are 28x28, 1 channel (greyscale)
    ),

    # Another convolution layer. This creates more flexibility in expressing non-linear transformations without losing information,
    # as found here: https://stackoverflow.com/questions/46515248/intuition-behind-stacking-multiple-conv2d-layers-before-dropout-in-cnn
    tf.keras.layers.Conv2D(
        32,                           # Amount of filters this layer uses. Creates a depth of x
        5,                            # Size of the nxn filter, use one value to use it for all spatial dimensions
        1,                            # Stride
        padding = 'valid',            # Use no padding
    ),

    # https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
    # Some of the features coming out of a Conv2D might be negative and might be truncated by a non-linearity like ReLU
    # If we normalize before activation we are including these negative values in the normalization immediately, 
    # before culling them from the feature space.
    tf.keras.layers.BatchNormalization(),

    # Pooling layer to achieve some translation invariance, as well as reduce resolution
    # Input will be of size 26x26x5, output will be 13x13x5
    tf.keras.layers.MaxPool2D(
        (2,2),            # Kernel size
        2,                # Stride
        padding = 'valid', # No padding
    ),
    
    # Dropout layer which will drop % of the inputs during training
    # Does not alter the input shape
    # Add randomization to prevent overfitting 
    tf.keras.layers.Dropout(.25),

    # Extract higher level features
    tf.keras.layers.Conv2D(
        5,
        5,
        1,
        activation = 'relu', # Use relu activation here
        padding = 'valid'
    ),

    # Introduce more randomisation to prevent overfitting
    tf.keras.layers.Dropout(.25),
    
    #  Will change shape from 13x13x
    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid'
    ),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer
    tf.keras.layers.Dense(10)
])

print("-- Compiling base model")
base_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("-- Fitting base model")
# We split the training data into a training set and validation set with 80-20% split
base_model.fit(
    train_images,
    train_labels,
    batch_size = 32,
    epochs=10,
    validation_split = .2,
    shuffle = True,
    callbacks = [
        callback.json_logging_callback,
        callback.print_callback,
        callback.lr_scheduler_callback
        ]
)

print("-- Evaluating base model")
test_loss, test_acc = base_model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([base_model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    util.plot_image(i, predictions[i], test_labels, test_images, class_names)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    util.plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
