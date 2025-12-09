# EEGNetv4
# Reproduced from https://github.com/vlawhern/arl-eegmodels
# Original paper: V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance,
#   “EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces,” in Journal of Neural Engineering, vol. 15, 2018, doi: 10.1088/1741-2552/aace8c.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

K.set_image_data_format("channels_first")

from .ftdropblock import FTDropBlock2D

def EEGNetv4(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropType='Dropout',block=15):
    
    input1 = Input(shape=(1, Chans, Samples))
    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = FTDropBlock2D(dropoutRate, block_size=block, tensorformat='NCHW')(block1, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(dropoutRate, data_format='channels_first')(block1, training=True) if dropType == 'SpatialDropout2D' else Dropout(dropoutRate)(block1, training=True)
        
    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = FTDropBlock2D(dropoutRate, block_size=block, tensorformat='NCHW')(block2, training=True) if dropType == 'FTDropBlock2D' else SpatialDropout2D(dropoutRate, data_format='channels_first')(block2, training=True) if dropType == 'SpatialDropout2D' else Dropout(dropoutRate)(block2, training=True)

    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten) 
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax, name='EEGNetv4')