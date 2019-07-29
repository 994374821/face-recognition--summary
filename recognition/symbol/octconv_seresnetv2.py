#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:49:33 2019

@author: gaomingda
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import symbol_resnetv2
import symbol_resnetv3
from config import config

import mxnet as mx

                
def get_before_pool(depth, ratio=-1, dropout=0., use_fp16=False):
    out = symbol_resnetv2.get_before_pool(depth=depth,
                                          ratio=ratio,
                                          dropout=dropout,
                                          use_fp16=use_fp16,
                                          use_se=True)
    return out

def get_linear(num_classes, depth, ratio=-1, dropout=0., use_fp16=False):
    out = symbol_resnetv2.get_linear(num_classes=num_classes, 
                                     depth=depth,
                                     ratio=ratio,
                                     dropout=dropout,
                                     use_fp16=use_fp16,
                                     use_se=True)
    return out

#def get_symbol(num_classes, depth, ratio=-1, dropout=0., use_fp16=False):
#    out = symbol_resnetv2.get_symbol(num_classes=num_classes, 
#                                     depth=depth,
#                                     ratio=ratio,
#                                     dropout=dropout,
#                                     use_fp16=use_fp16,
#                                     use_se=True)
#    return out

def get_symbol():

    num_classes = config.emb_size
    # depth = [26, 50, 101, 152, 200]
    depth = config.depth
    ratio = config.ratio
    bn_mom = config.bn_mom
    dropout = 0.4
    unit_v = config.net_unit
    
    # out = symbol_resnetv2.get_linear(num_classes=num_classes,
    #                                  depth=depth,
    #                                  ratio=ratio,
    #                                  dropout=0.4,
    #                                  use_fp16=False,
    #                                  use_se=False)
    # fc1 = mx.sym.BatchNorm(data=out, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')

    # (num_classes, depth, group=1, scaler=1., ratio=-1, dropout=0., use_fp16=False, use_se=False)
    if unit_v == 2:
        before_pool = symbol_resnetv2.get_before_pool(depth=depth,
                                                      group=1,
                                                      scaler=1,
                                                      ratio=ratio,
                                                      use_fp16=False,
                                                      use_se=False)
        if config.MP:
            before_pool2 = symbol_resnetv2.get_before_pool(depth=config.depth2,
                                                      group=1,
                                                      scaler=1,
                                                      ratio=ratio,
                                                      use_fp16=False,
                                                      use_se=False)
    elif unit_v == 3:
        before_pool = symbol_resnetv3.get_before_pool(depth=depth,
                                                      group=1,
                                                      scaler=1,
                                                      ratio=ratio,
                                                      use_fp16=False,
                                                      use_se=False)
    # - - - - -

    fc1 = mx.symbol.FullyConnected(data=before_pool, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    # plot network architecture
    digraph = mx.viz.plot_network(fc1, shape={'data': (1,3,112,112)}, save_format='pdf', node_attrs={"shape":"oval","fixedsize":"false"})
    digraph.render(filename='debug_octconv_res{}_3x3_se'.format(depth))
    return fc1


# code for debugging and plot network architecture
if __name__ == '__main__':

    # settings
    depth = [26, 50, 101, 152, 200][0]
    ratio = [-1, 0.125, 0.25, 0.5, 0.75][0] # set -1 for baseline network
    data_shape = (1, 3, 224, 224)
    
    # settings
    sym = get_linear(num_classes=1000, depth=depth, ratio=ratio)
    sym.save('symbol-debug.json')
    
    # print on terminal
    mx.visualization.print_summary(sym, shape={'data': data_shape})
    
    # plot network architecture
    digraph = mx.viz.plot_network(sym,  shape={'data': data_shape}, save_format='png')
    digraph.render(filename='debug')
    


