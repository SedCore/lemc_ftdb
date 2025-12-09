from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K



def EEGNetv4_DUQ(Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16):
    K.set_image_data_format("channels_first")

    input1 = Input(shape=(1, Chans, Samples))
    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    # Removed Dropout here
        
    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    # Removed Dropout here

    flatten = Flatten(name='flatten')(block2)
    # Removed Dense here
    # Removed Softmax here

    return Model(inputs=input1, outputs=flatten, name='EEGNetv4_DUQ')



n_ff = [2,4,8]
n_sf = [1,1,1]

def EEGITNet_DUQ(Chans=22, Samples=1125):
    K.set_image_data_format("channels_last")

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

    block_in = AveragePooling2D((1, 4))(block)
    # Removed Dropout here

    #================================

    block = ZeroPadding2D(padding=((3,0),(0,0)), data_format = "channels_first")(block_in)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    # Removed Dropout here
    block = ZeroPadding2D(padding=((3,0),(0,0)), data_format = "channels_first")(block)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    # Removed Dropout here
    block_out = Add()([block_in, block])


    block = ZeroPadding2D(padding=((6,0),(0,0)), data_format = "channels_first")(block_out)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    # Removed Dropout here
    block = ZeroPadding2D(padding=((6,0),(0,0)), data_format = "channels_first")(block)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    # Removed Dropout here
    block_out = Add()([block_out, block])


    block = ZeroPadding2D(padding=((12,0),(0,0)), data_format = "channels_first")(block_out)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    # Removed Dropout here
    block = ZeroPadding2D(padding=((12,0),(0,0)), data_format = "channels_first")(block)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    # Removed Dropout here
    block_out = Add()([block_out, block])


    block = ZeroPadding2D(padding=((24,0),(0,0)), data_format = "channels_first")(block_out)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    # Removed Dropout here
    block = ZeroPadding2D(padding=((24,0),(0,0)), data_format = "channels_first")(block)
    block = DepthwiseConv2D((1,4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    # Removed Dropout here
    block_out = Add()([block_out, block]) 

    #================================

    block = block_out

    #================================

    block = Conv2D(14, (1,1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = AveragePooling2D((4,1), data_format='Channels_first')(block)
    # Removed Dropout here
    embedded = Flatten()(block)

    # Removed Dense + Softmax here

    return Model(inputs=Input_block, outputs=embedded, name='EEGITNet_DUQ')