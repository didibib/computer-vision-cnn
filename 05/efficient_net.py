from tensorflow import keras
from tensorflow.keras import layers

# https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5
def bottleneck_block(filters, filter_size=(3,3), s = 1):
    module = [ layers.Conv2D(filters, (1,1)),
               layers.BatchNormalization(),
            #  keras.Activation('relu6'),
               layers.DepthwiseConv2D(filter_size, strides = (s,s)),
               layers.BatchNormalization(),
            #  keras.Activation('relu6'),
               layers.Conv2D(filters, (1,1)),
               layers.BatchNormalization()]
    return module

_efficient_net = [
    [layers.Conv2D(32, (3,3), strides = (2,2))],
    bottleneck_block(16),
    
    bottleneck_block(24, s = 2),
    bottleneck_block(24),
    
    bottleneck_block(24, (5,5), 2),
    bottleneck_block(24, (5,5)),

    bottleneck_block(80, s = 2),
    bottleneck_block(80),
    bottleneck_block(80),

    bottleneck_block(112, (5,5)),
    bottleneck_block(112, (5,5)),
    bottleneck_block(112, (5,5)),

    bottleneck_block(192, (5,5), 2),
    bottleneck_block(192, (5,5)),
    bottleneck_block(192, (5,5)),
    bottleneck_block(192, (5,5)),

    bottleneck_block(192),

    [layers.Conv2D(1280, (1,1)),
    layers.AvgPool2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(40)]
]

def efficient_net():
    model = keras.Sequential()
    for i in _efficient_net:
        for j in i:
            model.add(j)
    return model





