from tensorflow import keras

# https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5
def bottleneck_block(filters, filter_size=(3,3), s = 1):
    module = [ keras.Conv2D(filters, (1,1)),
               keras.BatchNormalization(),
            #  keras.Activation('relu6'),
               keras.DepthwiseConv2D(filter_size, stride = s),
               keras.BatchNormalization(),
            #  keras.Activation('relu6'),
               keras.Conv2D(filters, (1,1)),
               keras.BatchNormalization()]
    return module

efficient_net = keras.Sequential([
    keras.Conv2D(32, (3,3), stride = 2),
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

    keras.Conv2D(1280, (1,1)),
    keras.AvgPooling2D(),
    keras.Dense(256, activation='relu'),
    keras.Dense(40)
])





