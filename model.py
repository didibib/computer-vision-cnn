import tensorflow as tf
import matplotlib.pyplot as plt
import util
import json

class Model:
    name = 0
    class_names = 0

    _model = 0
    _json_log = 0
    _json_logging_callback = 0
    _lr_scheduler_callback = 0

    def __init__(self, model, class_names, name):
        self.name = name
        self.class_names = class_names
        self._model = model
        self._compile()

        self._lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(self._scheduler)

        self._json_logging_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: self._json_log.write(
            json.dumps({
                'epoch': epoch,
                'valid_loss': logs['val_loss'],
                'valid_acc': logs['val_accuracy'],
                'train_loss': logsð['loss'],
                'train_acc': logs['accuracy']
                }) + ',\n')
            )

    def run(self, train_images, train_labels, test_images, test_labels, draw = False):
        # Train the model
        self._json_log = open(self.name + '_epoch_log.json', mode='wt', buffering=1)
        self._fit(train_images, train_labels)

        # Test it's performance on the test images
        test_loss, test_acc = self._evaluate(test_images, test_labels)
        print('\nTest accuracy:', test_acc)
        
        self._json_log.write(json.dumps({'test_acc': test_acc, 'test_loss': test_loss}))
        probability_model = self._prob_model()
        predictions = probability_model.predict(test_images)
        if(draw):
            self._draw(test_images, test_labels, predictions)
        self._json_log.close()
        

    def _compile(self):
        self._model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    def _fit(self, train_images, train_labels):
        # We split the training data into a training set and validation set with 80-20% split
        self._model.fit(
            train_images,
            train_labels,
            batch_size = 32,
            epochs=10,
            verbose=2,
            validation_split = .2,
            shuffle = True,
            callbacks = [
                self._json_logging_callback,
                self._lr_scheduler_callback
            ]
        )
    
    def _evaluate(self, test_images, test_labels): 
        return self._model.evaluate(test_images,  test_labels, verbose=2)
    
    def _prob_model(self):
        return tf.keras.Sequential([self._model, tf.keras.layers.Softmax()])

    def _scheduler(self, epoch, lr):
        if epoch > 0 and epoch % 5 != 0:
            return lr
        else:
            return lr * .5

    def _draw(self, test_images, test_labels, predictions):
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            util.plot_image(i, predictions[i], test_labels, test_images, self.class_names)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            util.plot_value_array(i, predictions[i], test_labels)
        plt.tight_layout()
        plt.show()
        

base = tf.keras.Sequential([
    # We don't use an input layer since we specify the input_shape in the next layer (conv2D in this case)
    # Our input is of size 28x28x1, our output will be 24x24x32, since we have no padding and use 32 filters
    tf.keras.layers.Conv2D(
        32,                           # Amount of filters this layer uses. Creates a depth of x
        5,                            # Size of the nxn filter, use one value to use it for all spatial dimensions
        1,                            # Stride
        padding = 'same',             # Use no padding
        activation = 'relu',
        input_shape = (28, 28, 1) # Batch size of 32, Images are 28x28, 1 channel (greyscale)
    ),

    # Another convolution layer. This creates more flexibility in expressing non-linear transformations without losing information,
    # as found here: https://stackoverflow.com/questions/46515248/intuition-behind-stacking-multiple-conv2d-layers-before-dropout-in-cnn
    tf.keras.layers.Conv2D(
        32,                           # Amount of filters this layer uses. Creates a depth of x
        5,                            # Size of the nxn filter, use one value to use it for all spatial dimensions
        1,                            # Stride
        padding = 'valid',            # Use no padding
        activation = 'relu'
    ),

    # Pooling layer to achieve some translation invariance, as well as reduce resolution
    # Input will be of size 26x26x5, output will be 13x13x32
    tf.keras.layers.MaxPool2D(
        (2,2),            # Kernel size
        2,                # Stride
        padding = 'valid', # No padding
    ),
    
    # Dropout layer which will drop 25% of the inputs during training
    # Does not alter the input shape
    # Add randomization to prevent overfitting 
    tf.keras.layers.Dropout(.25),

    # Extract higher level features
    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'same',
        activation = 'relu',
    ),

    # Again, we stack our convolation layers to get more non-linear flexibility
    tf.keras.layers.Conv2D(
        32,
        5,
        1,
        padding = 'valid',
        activation = 'relu', # Use relu activation here
    ),

    # Introduce more randomisation to prevent overfitting
    tf.keras.layers.Dropout(.25),
    
    #  Will change shape from 13x13x32 to 6x6x32
    tf.keras.layers.MaxPool2D(
        (2,2),
        2,
        padding = 'valid',
    ),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer
    tf.keras.layers.Dense(10)
])