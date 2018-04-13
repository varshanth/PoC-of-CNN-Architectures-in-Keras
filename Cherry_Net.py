'''
File: Cherry_Net.py
Title: Cherry Net CNN Architecture
Description: Proof-of-Concept implementation of Cherry Net
'''


from keras.layers import Conv2D, Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
from ArchLayers import (_create_input_layer, _dense_transition, _cherry_layer,
                        _preactivation_layers)


def cherry_net(_input_shape, _n_classes):
    '''
    Expected Image Size for CIFAR10: 32x32xd (channel last)
            |
            v
    Conv2D: 3x3 D_OUT= D
            |
            v
    Cherry_Block0: {
        [(D_IN -> (1x1xD_IN, GROUP(3x3xDENSE_OUT)) -> D_OUT= D_IN + (G_RATE * NUM_DENSE))]  * (LAYERS_PER_BLOCK=2)
        [(D_IN -> (1x1xD_IN, GROUP(3x3xDENSE_OUT)) -> D_OUT= D_IN + (G_RATE * NUM_DENSE))]  * (LAYERS_PER_BLOCK=2)
        Dense_Transition:
        IN: (NxN, D_IN = D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/2 x N/2, D_OUT=COMPRESS_D0)
            |
            v    
    Cherry_Block1: {
        [(D_IN -> (1x1xD_IN, GROUP(3x3xDENSE_OUT)) -> D_OUT= D_IN + (G_RATE * NUM_DENSE))]  * (LAYERS_PER_BLOCK=2)
        [(D_IN -> (1x1xD_IN, GROUP(3x3xDENSE_OUT)) -> D_OUT= D_IN + (G_RATE * NUM_DENSE))]  * (LAYERS_PER_BLOCK=2)
        Dense_Transition:
        IN: (N/2xN/2, D_IN = D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/4 x N/4, D_OUT=COMPRESS_D1)
            |
            v    
    Cherry_Block2: {
        [(D_IN -> (1x1xD_IN, GROUP(3x3xDENSE_OUT)) -> D_OUT= D_IN + (G_RATE * NUM_DENSE))]  * (LAYERS_PER_BLOCK=2)
        [(D_IN -> (1x1xD_IN, GROUP(3x3xDENSE_OUT)) -> D_OUT= D_IN + (G_RATE * NUM_DENSE))]  * (LAYERS_PER_BLOCK=2)
            |
            v    
    Conv2D: D_IN -> (1x1xCOMPRESS_D2) -> D_OUT = COMPRESS_D2
            |
            v    
    GlobalAveragePool: OUT: 1 x 1, D_OUT = COMPRESS_D2
            |
            v
    Dense+Softmax Activation: D_OUT = N_CLASSES
            |
            v
    '''
    print('Architecture: Cherry Network')
    _d_init = 128 # Initial Depth
    _C = 8 # Cardinality
    _n_cherry_blocks = 3 # Number of spore blocks
    # Number of dense layers per cherry block
    _n_dense_layers_per_cherry_block = 2
    _n_cherry_layers_per_block = 2 # Number of Cherry Layers per block
    compress_factor = 1 # Compress Factor during Transition
    # Growth Rate for dense layers
    growth_rate = (_d_init // (_C * _n_dense_layers_per_cherry_block))
    _growth_rate_mul = 2 # Growth Rate Multiplier
    _max_d_out = 256 # Max Depth Out of Transition
    input_layer = _create_input_layer(_input_shape)
    intermed = Conv2D(_d_init, kernel_size=(3,3), padding='same')(input_layer)
    # Always new_depth must be divisible by _C (cardinality)
    new_depth = _d_init
    '''
    Output Sizes:
    *************
    Input Size = 32 x 32
    Output Size After ith Dense Transition to Cherry Block:
        1: 16 x 16
        2: 8 x 8
        
    Depth Sizes:
    ************
    Depth In = 64 
        1: 64 (64-> Cherry Layers ->256)
        2: 256 (256-> Cherry Layers -> 1024 -> Dense Transition ->256)
        3: 512 (256-> Cherry Layers -> 1024)
    Compress: 1024 -> ID -> 256
    Depth Out = 256
    '''
    for cherry_block_idx in range(_n_cherry_blocks):
        print('Block IDX = {0}'.format(cherry_block_idx))
        # Dense Block
        for cherry_layer_idx in range(_n_cherry_layers_per_block):
            intermed = _cherry_layer(intermed, _C, new_depth, growth_rate,
                                    _n_dense_layers_per_cherry_block)
            growth_rate = int(growth_rate * _growth_rate_mul)
            new_depth *= 2
        # Dynamic Compress Factor: No compression until depth exceeds Max_D_OUT
        compress_factor = min(1, _max_d_out/new_depth)
        # Compression Factor based growth rate
        growth_rate = (growth_rate if compress_factor == 1 else
                       (_max_d_out // (_C * _n_dense_layers_per_cherry_block)))
        if cherry_block_idx != (_n_cherry_blocks-1):
            # Dense Transition for every block exept last
            intermed = _dense_transition(intermed, new_depth,
                                        int(compress_factor * new_depth))
        new_depth = int(new_depth * compress_factor)
    
    intermed = _preactivation_layers(intermed)
    intermed = Conv2D(_max_d_out, kernel_size=(1,1))(intermed)
    gap_out = GlobalAveragePooling2D()(intermed)
    final_out = Dense(_n_classes, activation='softmax')(gap_out)
    model = Model(inputs=input_layer, outputs=final_out, name='Cherry_Net')
    return model


if __name__ == '__main__':
    print('\Cherry Net CNN')
    print('*********************\n')
    model = cherry_net((32,32,3),10)
    model.summary()
    save = int(input('Save Model Visualization to file? 0|1\n'))
    if save > 0:
        plot_model(model, to_file='{0}.png'.format(model.name),
                   show_shapes=True)