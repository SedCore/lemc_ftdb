# EEG-ITNet
# Reproduced from: https://github.com/AbbasSalami/EEG-ITNet
# Original paper: A. Salami, J. Andreu-Perez and H. Gillmeister,
#   "EEG-ITNet: An Explainable Inception Temporal Convolutional Network for motor imagery classification," in IEEE Access, doi: 10.1109/ACCESS.2022.3161489.

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D , SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, AveragePooling2D, Activation, Dropout, BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import Lambda, Add, Concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import SpatialDropout2D

from tensorflow.keras import backend as K
K.set_image_data_format("channels_last")

from .ftdropblock import FTDropBlock2D

n_ff = [2,4,8]
n_sf = [1,1,1]

def EEGITNet(out_class, Chans=22, Samples=1125, drop_rate=0.4, dropType='Dropout', blocksize=15):

    Input_block = Input(shape = (Chans, Samples, 1))

    block1 = Conv2D(n_ff[0], (1, 16), use_bias = False, activation = 'linear', padding='same', name = 'Spectral_filter_1')(Input_block)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias = False, padding='valid', depth_multiplier = n_sf[0], activation = 'linear',
                                 depthwise_constraint = max_norm(max_value=1), name = 'Spatial_filter_1')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)

    #================================

    block2 = Conv2D(n_ff[1], (1, 32), use_bias = False, activation = 'linear', padding='same', name = 'Spectral_filter_2')(Input_block)
    block2 = BatchNormalization()(block2)
    block2 = DepthwiseConv2D((Chans, 1), use_bias = False, padding='valid', depth_multiplier = n_sf[1], activation = 'linear',
                                 depthwise_constraint = max_norm(max_value=1), name = 'Spatial_filter_2')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)

    #================================

    block3 = Conv2D(n_ff[2], (1, 64), use_bias = False, activation = 'linear', padding='same', name = 'Spectral_filter_3')(Input_block)
    block3 = BatchNormalization()(block3)
    block3 = DepthwiseConv2D((Chans, 1), use_bias = False, padding='valid', depth_multiplier = n_sf[2], activation = 'linear',
                                 depthwise_constraint = max_norm(max_value=1), name = 'Spatial_filter_3')(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)

    #================================

    block = Concatenate(axis = -1)([block1, block2, block3]) 

    #================================

    block = AveragePooling2D((1, 4))(block)
    block_in = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)

    #================================

    block = ZeroPadding2D(padding=((3,0),(0,0)), data_format = "channels_first")(block_in)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)
    block = ZeroPadding2D(padding=((3,0),(0,0)), data_format = "channels_first")(block)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)
    block_out = Add()([block_in, block])


    block = ZeroPadding2D(padding=((6,0),(0,0)), data_format = "channels_first")(block_out)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)
    block = ZeroPadding2D(padding=((6,0),(0,0)), data_format = "channels_first")(block)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)
    block_out = Add()([block_out, block])


    block = ZeroPadding2D(padding=((12,0),(0,0)), data_format = "channels_first")(block_out)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)
    block = ZeroPadding2D(padding=((12,0),(0,0)), data_format = "channels_first")(block)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)
    block_out = Add()([block_out, block])


    block = ZeroPadding2D(padding=((24,0),(0,0)), data_format = "channels_first")(block_out)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)
    block = ZeroPadding2D(padding=((24,0),(0,0)), data_format = "channels_first")(block)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)
    block_out = Add()([block_out, block]) 

    #================================

    block = block_out

    #================================

    block = Conv2D(14, (1,1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = AveragePooling2D((4,1), data_format='Channels_first')(block)
    block = FTDropBlock2D(drop_rate, block_size=blocksize, tensorformat='NHWC')(block, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(drop_rate, data_format='channels_first')(block, training=True) if dropType == 'SpatialDropout2D' else Dropout(drop_rate)(block, training=True)
    embedded = Flatten()(block)

    out = Dense(out_class, activation = 'softmax', kernel_constraint = max_norm(0.25))(embedded)

    return Model(inputs=Input_block, outputs=out, name='EEGITNet')