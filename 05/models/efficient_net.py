from tensorflow import keras
from tensorflow.keras import layers

# https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5
# https://towardsdatascience.com/creating-deeper-bottleneck-resnet-from-scratch-using-tensorflow-93e11ff7eb02
# def bottleneck_block(filters, filter_size=(3,3), stride = 1):
#     block = keras.Sequential()
#     block.add(layers.Conv2D(filters, 1, padding = 'same'))
#     block.add(layers.BatchNormalization())
#     block.add(layers.Activation('relu6'))
#     block.add(layers.DepthwiseConv2D(filter_size, strides = (stride, stride), padding = 'same'))
#     block.add(layers.BatchNormalization())
#     block.add(layers.Activation('relu6'))
#     block.add(layers.Conv2D(filters, 1, padding = 'same'))
#     block.add(layers.BatchNormalization())
#     return block

def bottleneck_block(squeeze = 16, expand = 64):
    block = keras.Sequential()
    block.add(layers.Conv2D(expand, (1,1), activation='relu'))
    block.add(layers.DepthwiseConv2D((3,3), activation='relu'))
    block.add(layers.Conv2D(squeeze, (1,1)))
    return block

# def bot_block(expand=64, squeeze=16):
#     block = keras.Sequential()
#     block.add(layers.Conv2D(expand, (1,1), padding = 'same'))
#     block.add(layers.BatchNormalization())
#     block.add(layers.Activation('relu'))
#     block.add(layers.DepthwiseConv2D((3,3), padding = 'same'))
#     block.add(layers.BatchNormalization())
#     block.add(layers.Activation('relu'))
#     block.add(layers.Conv2D(squeeze, (1,1), padding = 'same'))
#     block.add(layers.BatchNormalization())
#     return block

def b0():
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3,3), strides = (2,2), padding = 'same'))
    
    model.add(bottleneck_block())
    model.add(layers.MaxPool2D(strides = 2))

    model.add(bottleneck_block())
    model.add(layers.MaxPool2D(strides = 2))

    model.add(bottleneck_block()) 
    model.add(layers.MaxPool2D(strides = 2))
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(80, activation = 'relu'))
    model.add(layers.Dense(40))

    return model

