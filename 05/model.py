from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import util
import json

class Model:
    name = 0
    class_names = 0

    _model = 0
    _logging_callback = 0
    _scheduler_callback = 0
    _log = []

    def __init__(self, model, class_names, name):
        self.name = name
        self.class_names = class_names
        self._model = model
        self._compile()

        self._scheduler_callback = keras.callbacks.LearningRateScheduler(self._scheduler)
        self._logging_callback = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: self._log.append(
                ('train',
                {'epoch': epoch,
                'val_loss': logs['val_loss'],
                'val_acc': logs['val_accuracy'],
                'train_loss': logs['loss'],
                'train_acc': logs['accuracy']})))

    def run(self, train_images, train_labels, test_images, test_labels, draw = False):
        # Train the model
        self._fit(train_images, train_labels)
        self._save_model()
        # Test it's performance on the test images
        test_loss, test_acc = self._evaluate(test_images, test_labels)
        print('\nTest accuracy:\n', test_acc)
        self._log.append(('test', { 'test_acc': test_acc, 'test_loss': test_loss}))
        #self._dump_log()

        # Probability
        # probability_model = self._prob_model()
        # predictions = probability_model.predict(test_images)
        # if(draw):
        #     self._draw(test_images, test_labels, predictions)
        
    # We assume everyting is in the right order
    def _dump_log(self):        
        json_log = open('json/' + self.name + '_epoch_log.json', mode='wt', buffering=1)
        json_log.write("{\"train_data\" : [\n")
        for i in range(len(self._log)):
            key, value = self._log[i]
            if key == 'train':
                # Dump entry
                json_log.write(json.dumps(value))
                next_key, _ = self._log[i+1]
                if next_key == 'train':
                    json_log.write(",\n")
            elif key == 'test':
                # Dump test data
                json_log.write("],\n")
                json_log.write("\"test_data\" :")
                json_log.write(json.dumps(value))
        json_log.write('}')
        json_log.close()
        self._log.clear()

    def _compile(self):
        self._model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    def _fit(self, train_images, train_labels):
        # We split the training data into a training set and validation set with 80-20% split
        split = -1 * int(len(train_images) * 0.2)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images[:split], train_labels[:split]))
        train_dataset = train_dataset.batch(32)

        val_dataset = tf.data.Dataset.from_tensor_slices((train_images[split:], train_labels[split:]))
        val_dataset = val_dataset.batch(32)
        self._model.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs=10,
            verbose=1,
            callbacks = [
                self._logging_callback
                # self._scheduler_callback
            ])
        
    def _save_model(self):
        self._model.save('models/'+ self.name)
   
    def _evaluate(self, test_images, test_labels): 
        return self._model.evaluate(test_images,  test_labels, verbose=2)
    
    def _prob_model(self):
        return keras.Sequential([self._model, keras.layers.Softmax()])

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