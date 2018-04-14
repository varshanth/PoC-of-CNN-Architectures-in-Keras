'''
File: SelectArch.py
Title: Select CNN Architecture
Description:
    Display Model Summaries & Visualizations for Implemented Architectures
'''

from Agg_Res_Net import agg_res_net
from Dense_Net import dense_net
from Spore_Net import spore_net
from Cherry_Net import cherry_net

if __name__ == '__main__':
    _arch_fn = {
            0: agg_res_net,
            1: dense_net,
            2: spore_net,
            3: cherry_net
            }
    selec_display = '0: Agg Res Net 1: Dense Net 2: Spore Net 3: Cherry Net'

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

