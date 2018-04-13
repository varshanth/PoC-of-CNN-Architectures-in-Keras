'''
File: Spore_Net.py
Title: Spore Net CNN Architecture
Description: Proof-of-Concept implementation of Spore Net
'''

from keras.layers import Conv2D, Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
from ArchLayers import (_create_input_layer, _dense_layer, _dense_transition,
                        _agg_res_layer)


def spore_net(_input_shape, _n_classes):
    '''
    Expected Image Size for CIFAR10: 32x32xd (channel last)
            |
            v
    Conv2D: 3x3 D_OUT= D
            |
            v
    Dense_Block0: (D_IN = D->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=2)
    Dense_Transition:
        IN: (NxN, D_IN = Dense_Block0_D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/2 x N/2, D_OUT=COMPRESS_D0)
    Agg_Res_Block0: (D_IN = COMPRESS_D0 ->(1x1, 3x3, 1x1) -> D_OUT= 2 * COMPRESS_D0) * (LAYERS_PER_BLOCK=2)
            |
            v    
    Dense_Block1: (D_IN ->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=2)
    Dense_Transition:
        IN: (N/2 x N/2, D_IN = Dense_Block1_D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/4 x N/4, D_OUT=COMPRESS_D1)
    Agg_Res_Block1: (D_IN = COMPRESS_D1 ->(1x1, 3x3, 1x1) -> D_OUT= 2 * COMPRESS_D1) * (LAYERS_PER_BLOCK=2)
            |
            v    
    Dense_Block2: (D_IN ->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=2)
    Dense_Transition:
        IN: (N/4 x N/4, D_IN = Dense_Block2_D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/8 x N/8, D_OUT=COMPRESS_D2)
    Agg_Res_Block2: (D_IN = COMPRESS_D2->(1x1, 3x3, 1x1) -> D_OUT= 2 * COMPRESS_D2) * (LAYERS_PER_BLOCK=2)
            |
            v    
    GlobalAveragePool: OUT: 1 x 1, D_OUT = Agg_Res_Block2_D_OUT
            |
            v
    Dense+Softmax Activation: D_OUT = N_CLASSES
            |
            v
    '''
    print('Architecture: Spore Network')
    _d_init = 64 # Initial Depth
    _C = 8 # Cardinality
    _n_spore_blocks = 3 # Number of spore blocks
    _n_agg_layer_per_block = 2 # Number of Aggregate Residual Layers per Block
    _growth_limit = 512 # Growth Limit
    _max_d_out = 256
    _n_dense_layers_per_block = 2 # Number of Dense Layers per block
    compress_factor = 1 # Compress Factor during Transition
    _strides = 1 # Strides (2 = Halve the output size, 1 = Retain output size)
    growth_rate = 32 # Growth Rate for dense layers
    _growth_rate_mul = 2 # Growth Rate Multiplier
    
    input_layer = _create_input_layer(_input_shape)
    intermed = Conv2D(_d_init, kernel_size=(3,3), padding='same')(input_layer)
    # Always new_depth must be divisible by _C (cardinality)
    new_depth = _d_init
    '''
    Output Sizes:
    *************
    Input Size = 32 x 32
    Output Size After ith Dense Transition to Agg Res Block:
        1: 16 x 16
        2: 8 x 8
        3: 4 x 4
        
    Depth Sizes:
    ************
    Depth In = 64 
        1: 64 (64-> Dense Layers->Dense Transition-> 128-> Agg Res Layers ->256)
        2: 256 (256-> Dense Layers->Dense Transition-> 256-> Agg Res Layers ->512)
        3: 512 (512-> Dense Layers->Dense Transition-> 256-> Agg Res Layers ->512)
    Depth Out = 512
    '''
    for spore_block_idx in range(_n_spore_blocks):
        print('Block IDX = {0}'.format(spore_block_idx))
        # Dense Block
        for dense_layer_idx in range(_n_dense_layers_per_block):
            intermed = _dense_layer(intermed, new_depth, growth_rate,
                                    _growth_limit)
            new_depth += growth_rate
        # Dynamic Compress Factor: No compression until depth exceeds Max_D_OUT
        compress_factor = min(1, _max_d_out/new_depth)
        # Dense Transition
        intermed = _dense_transition(intermed, new_depth,
                                         int(compress_factor * new_depth))
        new_depth = int(new_depth * compress_factor)
        growth_rate = int(growth_rate * _growth_rate_mul)
        # Aggregrated Residual Block
        for agg_layer_idx in range(_n_agg_layer_per_block):
            intermed = _agg_res_layer(intermed, new_depth, new_depth * 2,
                                      _C, _strides)
        new_depth *= 2
            
    gap_out = GlobalAveragePooling2D()(intermed)
    final_out = Dense(_n_classes, activation='softmax')(gap_out)
    model = Model(inputs=input_layer, outputs=final_out, name='Spore_Net')
    return model


if __name__ == '__main__':
    print('\nSpore Net CNN')
    print('*********************\n')
    model = spore_net((32,32,3),10)
    model.summary()
    save = int(input('Save Model Visualization to file? 0|1\n'))
    if save > 0:
        plot_model(model, to_file='{0}.png'.format(model.name),
                   show_shapes=True)
