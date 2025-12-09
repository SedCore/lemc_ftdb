# EEG-ITNet
# Reproduced from: https://github.com/AbbasSalami/EEG-ITNet
# Original paper: A. Salami, J. Andreu-Perez and H. Gillmeister,
#   "EEG-ITNet: An Explainable Inception Temporal Convolutional Network for motor imagery classification," in IEEE Access, doi: 10.1109/ACCESS.2022.3161489.
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Layer, Conv2D , SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, AveragePooling2D, Activation, Dropout, BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import Lambda, Add, Concatenate
from tensorflow.keras.constraints import max_norm
#from tensorflow.keras import layers
from .spectral_normalization import SpectralNormalization
from .gaussian_process import RandomFeatureGaussianProcess
from tensorflow.keras import backend as K

#K.set_image_data_format("channels_last")

###################################################

def EEGNetv4(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, num_inducing=512):
    K.set_image_data_format("channels_first")

    input1 = Input(shape=(1, Chans, Samples))
    
    block1 = SpectralNormalization(Conv2D(F1, (1, kernLength), padding='same', use_bias=False))(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = SpectralNormalization(DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.)))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SpectralNormalization(SeparableConv2D(F2, (1, 16), use_bias=False, padding='same'))(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    # Apply spectral normalization to dense layers
    dense = SpectralNormalization(Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate)))(flatten)
    # Gaussian Process output layer
    output = RandomFeatureGaussianProcess(nb_classes, num_inducing=num_inducing)(dense) # Returns (logits, covariance matrix)

    #output = Activation('softmax', name='softmax')(output[0]) # Apply softmax to logits only

    return Model(inputs=input1, outputs=output, name='EEGNetv4')

###################################################

n_ff = [2,4,8]
n_sf = [1,1,1]

def EEGITNet(out_class, Chans=22, Samples=1125, drop_rate=0.4, num_inducing=512):

    Input_block = Input(shape = (Chans, Samples, 1))

    block1 = SpectralNormalization(Conv2D(n_ff[0], (1, 16), use_bias = False, activation = 'linear', padding='same', name = 'Spectral_filter_1'))(Input_block)
    block1 = BatchNormalization()(block1)
    block1 = SpectralNormalization(DepthwiseConv2D((Chans, 1), use_bias = False, padding='valid', depth_multiplier = n_sf[0], activation = 'linear', depthwise_constraint = max_norm(max_value=1), name = 'Spatial_filter_1'))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)

    #================================

    block2 = SpectralNormalization(Conv2D(n_ff[1], (1, 32), use_bias = False, activation = 'linear', padding='same', name = 'Spectral_filter_2'))(Input_block)
    block2 = BatchNormalization()(block2)
    block2 = SpectralNormalization(DepthwiseConv2D((Chans, 1), use_bias = False, padding='valid', depth_multiplier = n_sf[1], activation = 'linear', depthwise_constraint = max_norm(max_value=1), name = 'Spatial_filter_2'))(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)

    #================================

    block3 = SpectralNormalization(Conv2D(n_ff[2], (1, 64), use_bias = False, activation = 'linear', padding='same', name = 'Spectral_filter_3'))(Input_block)
    block3 = BatchNormalization()(block3)
    block3 = SpectralNormalization(DepthwiseConv2D((Chans, 1), use_bias = False, padding='valid', depth_multiplier = n_sf[2], activation = 'linear', depthwise_constraint = max_norm(max_value=1), name = 'Spatial_filter_3'))(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)

    #================================

    block = Concatenate(axis = -1)([block1, block2, block3]) 

    #================================

    block = AveragePooling2D((1, 4))(block)
    block_in = Dropout(drop_rate)(block)

    #================================

    block = ZeroPadding2D(padding=((3,0),(0,0)), data_format = "channels_first")(block_in)
    block = SpectralNormalization(DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1)))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = ZeroPadding2D(padding=((3,0),(0,0)), data_format = "channels_first")(block)
    block = SpectralNormalization(DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1)))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_in, block])


    block = ZeroPadding2D(padding=((6,0),(0,0)), data_format = "channels_first")(block_out)
    block = SpectralNormalization(DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2)))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = ZeroPadding2D(padding=((6,0),(0,0)), data_format = "channels_first")(block)
    block = SpectralNormalization(DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2)))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block])


    block = ZeroPadding2D(padding=((12,0),(0,0)), data_format = "channels_first")(block_out)
    block = SpectralNormalization(DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4)))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = ZeroPadding2D(padding=((12,0),(0,0)), data_format = "channels_first")(block)
    block = SpectralNormalization(DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4)))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block])


    block = ZeroPadding2D(padding=((24,0),(0,0)), data_format = "channels_first")(block_out)
    block = SpectralNormalization(DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8)))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = ZeroPadding2D(padding=((24,0),(0,0)), data_format = "channels_first")(block)
    block = SpectralNormalization(DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8)))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block]) 

    #================================

    block = block_out

    #================================

    block = SpectralNormalization(Conv2D(14, (1,1)))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = AveragePooling2D((4,1), data_format='Channels_first')(block)
    block = Dropout(drop_rate)(block)
    embedded = Flatten()(block)

    # Apply spectral normalization to the feature embedding
    embedded = SpectralNormalization(Dense(out_class, activation = 'linear', kernel_constraint = max_norm(0.25)))(embedded)

    # Gaussian Process output layer for uncertainty estimation
    output = RandomFeatureGaussianProcess(out_class, num_inducing=num_inducing)(embedded) # Returns (logits, covariance matrix)

    #output = Activation('softmax', name='softmax')(output[0]) # Apply softmax to logits only

    model = Model(inputs=Input_block, outputs=output, name='EEGITNet')
    return model


