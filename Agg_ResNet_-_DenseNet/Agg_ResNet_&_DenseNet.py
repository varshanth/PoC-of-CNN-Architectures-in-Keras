from keras.layers import (Conv2D, Dropout, BatchNormalization,
                          Input, Dense, Lambda, concatenate, add)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
import keras.backend as K
from keras.models import Model
from keras.utils.vis_utils import plot_model

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
    # AVERAGE_POOL
    pool_out = MaxPooling2D(pool_size=(2,2), strides=(2,2))(compress_in)
    return pool_out


def agg_res_net(_input_shape, _n_classes):
    '''
    Expected Image Size for CIFAR10: 32x32xd (channel last)
            |
            v    
    Conv2D: 3x3 D_OUT= D
            |
            v
    Agg_Res_Block0: (D_IN = 2D->(1x1, 3x3, 1x1) -> D_OUT=4D) * (LAYERS_PER_BLOCK=6)
            |
            v
    Agg_Res_Block1: (D_IN = 4D->(1x1, 3x3, 1x1) -> D_OUT=8D) * (LAYERS_PER_BLOCK=6)
            |
            v
    Agg_Res_Block2: (D_IN = 8D->(1x1, 3x3, 1x1) -> D_OUT=16D) * (LAYERS_PER_BLOCK=6)
            |
            v
    GlobalAveragePool: OUT: 1 x 1, D_OUT = 16D
            |
            v
    Dense+Softmax Activation: D_OUT = N_CLASSES
            |
            v
    '''
    print('Architecture: Aggregated Residual Network')
    _d_init = 32 # Initial Depth
    _C = 8 # Cardinality
    _n_agg_blocks = 3 # Number of Aggregate Residual Blocks
    _n_agg_layer_per_block = 6 # Number of Aggregate Residual Layers per Block
    
    input_layer = _create_input_layer(_input_shape)
    intermed = Conv2D(_d_init, kernel_size=(3,3), padding='same')(input_layer)
    new_depth = _d_init * 2
    '''
    Output Sizes:
    *************
    Input Size = 32 x 32
    Output Size After ith block:
        1: 16 x 16
        2: 8 x 8
        3: 4 x 4
        
    Depth Sizes:
    ************
    Depth to 1st Block = 64
    Depth to the ith (i > 1) block:
        2: 128
        3: 256
    Depth Out = 512
    '''
    for agg_block_idx in range(_n_agg_blocks):
        print('Block IDX = {0}'.format(agg_block_idx))
        for agg_layer_idx in range(_n_agg_layer_per_block):
            # Double the stride once per block to halve the output size
            strides = 2 if agg_layer_idx == 0 else 1
            # Can't add shorcut between input and downsampled input
            intermed = _agg_res_layer(intermed, new_depth, 2 * new_depth,
                                      _C, strides)
        '''
        Double the depth after a block to capture the bigger receptive field
        since each block involves 1 stride of 2
        '''
        new_depth *= 2
    gap_out = GlobalAveragePooling2D()(intermed)
    final_out = Dense(_n_classes, activation='softmax')(gap_out)
    model = Model(inputs=input_layer, outputs=final_out, name='Agg_Res_Net')
    return model


def dense_net(_input_shape, _n_classes):
    '''
    Expected Image Size for CIFAR10: 32x32xd (channel last)
            |
            v  
    Conv2D: 3x3 D_OUT= D
            |
            v
    Dense_Block0: (D_IN = D->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=4)
    Dense_Transition:
        IN: (NxN, D_IN = Dense_Block0_D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/2 x N/2, D_OUT=COMPRESS_D0)
            |
            v    
    Dense_Block1: (D_IN = D->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=4)
    Dense_Transition:
        IN: (N/2 x N/2, D_IN = Dense_Block1_D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/4 x N/4, D_OUT=COMPRESS_D1)
            |
            v    
    Dense_Block2: (D_IN = D->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=4)
    Dense_Transition:
        IN: (N/4 x N/4 , D_IN = Dense_Block2_D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/8 x N/8, D_OUT=COMPRESS_D2)
            |
            v
    Dense_Block3: (D_IN = D->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=4)
            |
            v    
    GlobalAveragePool: OUT: 1 x 1, D_OUT = Dense_Block3_D_OUT
            |
            v
    Dense+Softmax Activation: D_OUT = N_CLASSES
            |
            v
    '''
    print('Architecture: Dense Network')
    _d_init = 64 # Initial Depth
    _growth_limit = 376 #  Growth Limit
    _n_dense_blocks = 4 # Number of Dense Blocks
    _n_dense_layers_per_block = 4 # Number of Dense Layers per block
    _compress_factor = 1.0 # Compress Factor during Transition
    growth_rate = 16 # Growth Rate for dense layers
    _growth_rate_mul = 2 # Growth Rate Multiplier to be used between blocks
    
    input_layer = _create_input_layer(_input_shape)
    intermed = Conv2D(_d_init, kernel_size=(3,3), padding='same')(input_layer)
    new_depth = _d_init
    '''
    Output Sizes:
    *************
    Input Size = 32 x 32
    Output Size After ith Transition:
        1: 16 x 16
        2: 8 x 8
        3: 4 x 4
        
    Depth Sizes:
    ************
    Depth to 1st Block = 64
    Depth to the ith (i > 1) block:
        2: 128
        3: 256
        4: 512
    Depth Out = 1024
    '''
    for dense_block_idx in range(_n_dense_blocks):
        print('Block IDX = {0}'.format(dense_block_idx))
        for dense_layer_idx in range(_n_dense_layers_per_block):
            intermed = _dense_layer(intermed, new_depth, growth_rate,
                                    _growth_limit)
            new_depth += growth_rate
        if dense_block_idx != (_n_dense_blocks-1):
            # No Dense Transition for last layer
            intermed = _dense_transition(intermed, new_depth,
                                         int(_compress_factor * new_depth))
        # Impose Dynamic Growth Rate
        growth_rate = int(growth_rate * _growth_rate_mul)
        new_depth = int(_compress_factor * new_depth)
        
    gap_out = GlobalAveragePooling2D()(intermed)
    final_out = Dense(_n_classes, activation='softmax')(gap_out)
    model = Model(inputs=input_layer, outputs=final_out, name='Dense_Net')
    return model


if __name__ == '__main__':
    _arch_fn = {
            0:agg_res_net,
            1:dense_net
            }
    selec_display = '0: Agg Res Net 1: Dense Net'
    arch = int(input(
            '''
            ******************Choose Architecture ****************
            {0}
            ******************************************************
          '''.format(selec_display)))
    model = _arch_fn[arch]((32,32,3),10)
    model.summary()
    save = int(input('Save Model Visualization to file? 0|1\n'))
    if save > 0:
        plot_model(model, to_file='{0}.png'.format(model.name),
                   show_shapes=True)  
