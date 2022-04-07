# Standalone file to store the weights from each model
from tensorflow import keras

model = keras.models.load_model('models/base')
model.save_weights('weights/base.h5')

model = keras.models.load_model('models/variant1_dense')
model.save_weights('weights/variant1_dense.h5')

model = keras.models.load_model('models/variant2_kernel')
model.save_weights('weights/variant2_kernel.h5')

model = keras.models.load_model('models/variant3_lrelu')
model.save_weights('weights/variant3_lrelu.h5')

model = keras.models.load_model('models/variant4_batch_norm')
model.save_weights('weights/variant4_batch_norm.h5')