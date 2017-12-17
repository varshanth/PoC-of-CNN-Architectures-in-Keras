from keras.layers import Conv2D, Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
from ArchLayers import (_preactivation_layers, _create_input_layer,
        _agg_res_layer, _dense_layer, _dense_transition)


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


if __name__ == '__main__':
    print('\nAggregated Residual Transformations Network')
    print('********************************************\n')
    model = agg_res_net((32,32,3),10)
    model.summary()
    save = int(input('Save Model Visualization to file? 0|1\n'))
    if save > 0:
        plot_model(model, to_file='{0}.png'.format(model.name),
                   show_shapes=True)
