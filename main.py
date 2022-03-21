import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import util

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

# Shuffle training data using permutations
print("-- Shuffle training data")
train_images, train_labels = util.unison_shuffled_copies(train_images, train_labels)

# We go for an 80-20 split
# So 80% for our training, and 20% for our validation
train_size = 0.8
train_images, validate_images = np.split(train_images, [int(train_size*len(train_images))])
train_labels, validate_labels = np.split(train_labels, [int(train_size*len(train_labels))])

# Pooling layers : provide an approach to down sampling feature maps by summarizing 
# the presence of features in patches of the feature map. Two common pooling methods 

base_model = tf.keras.Sequential([
    # We don't use an input layer since we specify the input_shape in the next layer (conv2D in this case)
    # Our input is of size 28x28x1, our output will be 26x26x5, since we have no padding and use 5 filters
    tf.keras.layers.Conv2D(
        5,                            # Amount of filters this layer uses. Creates a depth of x
        5,                            # Size of the nxn filter, use one value to use it for all spatial dimensions
        1,                            # Stride
        padding = 'valid',            # Use no padding
        input_shape = (32, 28, 28, 1) # Batch size of 32, Images are 28x28, 1 channel (greyscale)
    ),
    tf.keras.layers.BatchNormalization(),
    # Pooling layer to achieve some translation invariance, as well as reduce resolution
    # Input will be of size 26x26x5, output will be 13x13x5
    tf.keras.layers.MaxPool2D(
        (2,2),            # Kernel size
        2,                # Stride
        padding = 'valid' # No padding
    ),
    # Dropout layer which will drop 10% of the inputs during training
    # Does not alter the input shape
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Conv2D(
        5,
        5,
        1,
        activation = 'relu', # Use relu activation here
        padding = 'valid'
    ),
    #  Will change shape from 13x13x
    tf.keras.layers.AvgPool2D(
        (2,2),
        2,
        padding = 'valid'
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer
    tf.keras.layers.Dense(10)
])

base_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

base_model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = base_model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([base_model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
