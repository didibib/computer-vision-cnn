from tensorflow import keras
from tensorflow.keras import layers

# https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5
# https://towardsdatascience.com/creating-deeper-bottleneck-resnet-from-scratch-using-tensorflow-93e11ff7eb02
def bottleneck_block(filters, filter_size=(3,3), stride = 1):
    block = keras.Sequential()
    block.add(layers.Conv2D(filters, 1))
    block.add(layers.BatchNormalization())
    block.add(layers.DepthwiseConv2D(filter_size, strides = (stride, stride)))
    block.add(layers.Conv2D(filters, 1))
    block.add(layers.BatchNormalization())
    return block

def efficient_net():
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3,3), strides = (2,2))),
    model.add(bottleneck_block(16)),

    model.add(bottleneck_block(24, stride = 2)),
    model.add(bottleneck_block(24)),

    model.add(bottleneck_block(24, (5,5), 2)),
    model.add(bottleneck_block(24, (5,5))),

    model.add(bottleneck_block(80, stride=2)),
    model.add(bottleneck_block(80))
    model.add(bottleneck_block(80))

    model.add(bottleneck_block(112, (5,5)))
    model.add(bottleneck_block(112, (5,5)))
    model.add(bottleneck_block(112, (5,5)))

    model.add(bottleneck_block(192, (5,5), 2)),
    model.add(bottleneck_block(192, (5,5))),
    model.add(bottleneck_block(192, (5,5))),
    model.add(bottleneck_block(192, (5,5))),

    model.add(bottleneck_block(192)),  

    model.add(layers.Conv2D(1280, (1,1))),
    model.add(layers.AvgPool2D()),
    model.add(layers.Dense(256, activation = 'relu')),
    model.add(layers.Dense(40))

    return model





