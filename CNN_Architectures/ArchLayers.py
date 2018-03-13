'''
File: ArchLayers.py
Title: CNN Architecture Layers Library
Description: Contains APIs for creating Architecture Specific CNN Layers
'''

from keras.layers import (Conv2D, Dropout, BatchNormalization,
                          Input, Dense, Lambda, concatenate, add)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
import keras.backend as K


_channel_axis = 3 if K.image_data_format() == 'channels_last' else 1


def _preactivation_layers(input_layer):
    '''
    Apply BatchNorm + Leaky ReLU
    '''
    BN = BatchNormalization()(input_layer)
    Activated = LeakyReLU()(BN)
    return Activated


def _create_input_layer(input_shape):
    '''
    Input Shape: (H, W, D) if channels last
    '''
    return Input(shape=input_shape)


def _agg_res_layer(input_layer, d_in, d_out, C, _strides):
    '''
    ID_IN: 1x1 Conv bottleneck to d_in
    CONV: 3x3 Conv with stride on each cardinal branch
    MERGE: Merge cardinal branches channel wise
    ID_OUT: 1x1 Conv expansion to d_out
    ID_EXPAND: if d_in != d_out, perform 1x1 Conv to d_out
    SHORTCUT: Add residual connection
    '''
    print('Aggregated Res Layer: d_in={0}'.format(d_in))
    # ID_IN
    activ_in = _preactivation_layers(input_layer)
    bottleneck = Conv2D(d_in, kernel_size=(1,1), padding='same')(activ_in)
    channels_per_branch = d_in // C
    c_blocks = []
    # CONV
    for c_block_idx in range(C):
        start_idx = c_block_idx * channels_per_branch
        end_idx = start_idx + channels_per_branch
        c_block = Lambda(
                lambda c_b: (c_b[:, :, :, start_idx:end_idx]
                if K.image_data_format() == 'channels_last' else
                c_b[:, start_idx:end_idx :, :,]))(bottleneck)
        c_block = _preactivation_layers(c_block)
        c_block = Conv2D(channels_per_branch, kernel_size=(3,3),
                strides=(_strides, _strides), padding='same')(c_block)
        c_blocks.append(c_block)
    # MERGE
    merged = concatenate(c_blocks, axis = _channel_axis)
    # ID_OUT
    out = _preactivation_layers(merged)
    out = Conv2D(d_out, kernel_size=(1,1), padding='same')(merged)
    input_layer = _preactivation_layers(input_layer)
    id_expand = Conv2D(d_out, kernel_size=(1,1),
                           strides=(_strides, _strides))(input_layer)
    # SHORTCUT
    resid_out = add([out, id_expand])
    return resid_out


def _dense_layer(input_layer, d_in, growth_rate, growth_limit):
    '''
    ID_IN: Bottleneck depth to growth limit
    COMPRESS_CONV: Perform Conv with depth = growth rate
    CONCAT_INPUT: Increase density
    '''
    print('Dense Layer: d_in = {0}'.format(d_in))
    # ID_IN
    bottleneck = None
    if d_in > growth_limit:
        bottleneck = _preactivation_layers(input_layer)
        bottleneck = Conv2D(growth_limit, kernel_size=(1,1),
                            padding='same')(bottleneck)
    # COMPRESS_CONV
    compress_in = _preactivation_layers(
            bottleneck if bottleneck != None else input_layer)
    dense_in = Conv2D(growth_rate, kernel_size=(3,3),
                      padding='same')(compress_in)
    # CONCAT_INPUT
    return concatenate([input_layer, dense_in], axis = _channel_axis)


def _dense_transition(input_layer, d_in, d_out):
    '''
    ID_IN: Compress input layer using 1x1 Convs
    MAX_POOL: Extract summary, halve the output size by using stride 2
    '''
    print('Dense Transition: d_in={0}'.format(d_in))
    input_layer = _preactivation_layers(input_layer)
    # ID_IN
    compress_in = Conv2D(d_out, kernel_size=(1,1), padding='same')(input_layer)
    # MAX_POOL
    pool_out = MaxPooling2D(pool_size=(2,2), strides=(2,2))(compress_in)
    return pool_out


if __name__ == '__main__':
    print('CNN Architecture Layers Library')


