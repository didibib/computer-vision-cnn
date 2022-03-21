import json
import tensorflow as tf

# Stream the epoch loss to a file in JSON format. The file content
# is not well-formed JSON but rather has a JSON object per line.
json_log = open('epoch_log.json', mode='wt', buffering=1)
json_logging_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss'], 'accuracy': logs['accuracy']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

print_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(
        '\nepoch: ', epoch, ' accuracy: ', logs['val_accuracy'], ' loss: ', logs['val_loss']
    )
)

def scheduler(epoch, lr):
  if epoch > 0 and epoch % 5 != 0:
    return lr
  else:
    return lr * .5

lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)