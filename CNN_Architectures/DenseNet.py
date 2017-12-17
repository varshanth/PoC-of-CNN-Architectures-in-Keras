'''
File: DenseNet.py
Title: Densely Connected CNN
Description: Proof-of-Concept implementation of DenseNet
'''

from keras.layers import Conv2D, Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
from ArchLayers import (_preactivation_layers, _create_input_layer,
        _agg_res_layer, _dense_layer, _dense_transition)


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
    print('\nDensely Connected CNN')
    print('*********************\n')
    model = dense_net((32,32,3),10)
    model.summary()
    save = int(input('Save Model Visualization to file? 0|1\n'))
    if save > 0:
        plot_model(model, to_file='{0}.png'.format(model.name),
                   show_shapes=True)
