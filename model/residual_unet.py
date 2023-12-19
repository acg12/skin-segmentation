from tensorflow import keras
import tensorflow as tf

AXIS = 3
SIZE = 128

def residual_conv_block(x, filter_size, size, dropout, batch_norm=False):
    conv = keras.layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = keras.layers.BatchNormalization(axis=AXIS)(conv)
    conv = keras.layers.Activation('relu')(conv)
    
    conv = keras.layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    
    if batch_norm is True:
        conv = keras.layers.BatchNormalization(axis=AXIS)(conv)
    conv = keras.layers.Activation('relu')(conv)

    if dropout > 0:
        conv = keras.layers.Dropout(dropout)(conv)

    shortcut = keras.layers.Conv2D(size, kernel_size=(1,1), padding='same')(x)

    residual_path = keras.layers.add([conv, shortcut])
    residual_path = keras.layers.Activation('relu')(residual_path)

    return residual_path

def Residual_UNet(input_shape=(SIZE, SIZE, 3), num_classes=1, dropout=0.0, batch_norm=True):
    NUMBER_OF_FILTER = 64
    FILTER_SIZE = 3
    UPSAMPLING_SIZE = 2

    inputs = keras.layers.Input(input_shape, dtype=tf.float32)

  # Downsampling Layers
    conv_1 = residual_conv_block(inputs, FILTER_SIZE, NUMBER_OF_FILTER,
                                dropout, batch_norm)
    pool_1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv_1)

    conv_2 = residual_conv_block(pool_1, FILTER_SIZE, 2*NUMBER_OF_FILTER,
                                  dropout, batch_norm)
    pool_2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv_2)

    conv_3 = residual_conv_block(pool_2, FILTER_SIZE, 4*NUMBER_OF_FILTER,
                                  dropout, batch_norm)
    pool_3 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv_3)

    conv_4 = residual_conv_block(pool_3, FILTER_SIZE, 8*NUMBER_OF_FILTER,
                                  dropout, batch_norm)
    pool_4 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv_4)

    conv_5 = residual_conv_block(pool_4, FILTER_SIZE, 16*NUMBER_OF_FILTER,
                              dropout, batch_norm)

  # Upsampling Layers
    up_4 = keras.layers.UpSampling2D(size=(UPSAMPLING_SIZE, UPSAMPLING_SIZE),
                                      data_format='channels_last')(conv_5)
    up_4 = keras.layers.concatenate([up_4, conv_4], axis=3)
    up_conv_4 = residual_conv_block(up_4, FILTER_SIZE, 8*NUMBER_OF_FILTER,
                                      dropout, batch_norm)

    up_3 = keras.layers.UpSampling2D(size=(UPSAMPLING_SIZE, UPSAMPLING_SIZE),
                                      data_format='channels_last')(up_conv_4)
    up_3 = keras.layers.concatenate([up_3, conv_3], axis=3)
    up_conv_3 = residual_conv_block(up_3, FILTER_SIZE, 4*NUMBER_OF_FILTER,
                                      dropout, batch_norm)

    up_2 = keras.layers.UpSampling2D(size=(UPSAMPLING_SIZE, UPSAMPLING_SIZE),
                                      data_format='channels_last')(up_conv_3)
    up_2 = keras.layers.concatenate([up_2, conv_2], axis=3)
    up_conv_2 = residual_conv_block(up_2, FILTER_SIZE, 2*NUMBER_OF_FILTER,
                                      dropout, batch_norm)

    up_1 = keras.layers.UpSampling2D(size=(UPSAMPLING_SIZE, UPSAMPLING_SIZE),
                                        data_format='channels_last')(up_conv_2)
    up_1 = keras.layers.concatenate([up_1, conv_1], axis=3)
    up_conv_1 = residual_conv_block(up_1, FILTER_SIZE, NUMBER_OF_FILTER,
                                  dropout, batch_norm)

  # 1*1 Convolutional Layers
    final_conv = keras.layers.Conv2D(num_classes, kernel_size=(1,1))(up_conv_1)
    final_conv = keras.layers.BatchNormalization(axis=3)(final_conv)
    final_conv = keras.layers.Activation('sigmoid')(final_conv)

  # Model Integration
    model = keras.models.Model(inputs, final_conv, name='ResUNet')
    return model