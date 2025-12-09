# FT-DropBlock regularization
# Attribution to: Pierre Sedi N.
# https://github.com/SedCore/FTDropBlock
# Original paper: P. Sedi Nzakuna, V. Paciello, V. Gallo, A. Lay-Ekuakille, A. Kuti Lusala,
#   “FT-DropBlock: A Novel Approach for SPatiotemporal Regularization in EEG-based Convolutional Neural Networks,”
#   in 2025 IEEE International Instrumentation and Measurement Technology Conference (I2MTC), 2025
#
# Inspired from: https://github.com/miguelvr/dropblock

import tensorflow as tf
from tensorflow.keras.layers import Layer, Permute
from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable

@register_keras_serializable()
class FTDropBlock2D(Layer):
    def __init__(self, drop_prob, block_size=4, tensorformat='NHWC', **kwargs):
        """
        FT-DropBlock for 2D EEG feature maps in Tensorflow Keras.

        Args:
            drop_prob (float): Probability of dropping blocks.
            block_size (int): Size of the block to drop.
            tensorformat (string): Format of the input tensor, either NHWC (channels last) or NCHW (channels first).
        """
        super(FTDropBlock2D, self).__init__(**kwargs)
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.tensorformat = tensorformat

    def build(self, input_shape):
        super(FTDropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None):
        if not training or self.drop_prob == 0.:
            return inputs
        else:
            # Reshape the tensor to a custom (N,C,W,H) format corresponding to (batch_size, features, time points, electrodes)
            if self.tensorformat == 'NHWC':
                inputs = Permute((3, 2, 1))(inputs) # convert into NCWH. The dimension N at index 0 remains unchanged.
            elif self.tensorformat == 'NCHW':
                inputs = Permute((1, 3, 2))(inputs) # convert into NCWH. The dimension N at index 0 remains unchanged.
                        
            input_shape = tf.shape(inputs)
            batch_size, features, time_points, electrodes = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

            # Get gamma value
            gamma = self._compute_gamma()

            # Sample mask
            mask = tf.cast(tf.keras.random.uniform((batch_size, features, time_points)) < gamma, tf.float32)
            
            # Compute block mask
            block_mask = self._compute_block_mask(mask, self.block_size, features, time_points)

            # Expand block mask to match input channels
            block_mask = tf.expand_dims(block_mask, axis=-1)
            block_mask = tf.tile(block_mask, [1, 1, 1, electrodes])

            # Apply block mask
            outputs = inputs * block_mask
            
            # Scale output
            outputs = outputs * tf.cast(tf.size(block_mask), tf.float32) / K.sum(block_mask)
            
            # Reshape the tensor back to its initial format
            if self.tensorformat == 'NHWC':
                outputs = Permute((3, 2, 1))(outputs)
            elif self.tensorformat == 'NCHW':
                outputs = Permute((1, 3, 2))(outputs)
            return outputs

    def _compute_block_mask(self, mask, block_size, height, width):
        # Pad the mask so that pooling will apply to edge regions
        pad = block_size // 2
        mask = tf.pad(mask, [[0, 0], [pad, pad], [pad, pad]], mode='CONSTANT', constant_values=0)

        # Use max pooling to create a block-wise mask in the spatial dimensions
        block_mask = tf.nn.max_pool2d(mask[:, :, :, tf.newaxis],
                                      ksize=[1, block_size, block_size, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

        block_mask = 1 - tf.squeeze(block_mask, axis=-1)

        # Crop back to the original size
        block_mask = block_mask[:, pad:pad + height, pad:pad + width]

        return block_mask

    def _compute_gamma(self):
        # Compute gamma value based on current drop probability and block size
        return self.drop_prob / (self.block_size ** 2)