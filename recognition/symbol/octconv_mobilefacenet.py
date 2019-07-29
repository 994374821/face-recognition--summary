import sys
import os
import mxnet as mx
import symbol_utils

import symbol_octconv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config


def Act(data, act_type, name):
    # ignore param act_type, set it in this function
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body


def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix='', ratio=-1.):
    # (data, num_filter, kernel, pad=(0, 0), stride=(1,1), name=None, no_bias=True, num_group=1, act_type='relu')
    if ratio>=0.:
        num_filter = (num_filter - int(ratio*num_filter), int(ratio*num_filter))
        if num_group>1:
            num_group = num_filter
    act = symbol_octconv.Conv_BN_ACT(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride,
                               pad=pad, no_bias=True, act_type=config.net_act, name='%s%s_conv2d' % (name, suffix))
    return act


def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix='', ratio=-1.):

    if ratio>=0.:
        num_filter = (num_filter - int(ratio*num_filter), int(ratio*num_filter))
        if num_group>1:
            num_group = num_filter
    bn = symbol_octconv.Conv_BN(data, num_filter, kernel, pad=pad, stride=stride, name='%s%s_conv2d' %(name, suffix),
                                no_bias=True, num_group=num_group, zero_init_gamma=False)

    return bn


def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix='', ratio=-1.):
    if ratio==-1.:
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride,
                                  pad=pad, no_bias=True, name='%s%s_conv2d' % (name, suffix))
    else:
        num_filter = (num_filter - int(ratio * num_filter), int(ratio * num_filter))
        if num_group>1:
            num_group = num_filter
        conv = symbol_octconv.Convolution(data, num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group,
                                   no_bias=False, name='%s%s_conv2d' % (name, suffix))

    return conv


def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix='', ratio=-1.):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                name='%s%s_conv_sep' % (name, suffix), ratio=ratio)
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride,
                   name='%s%s_conv_dw' % (name, suffix), ratio=ratio)
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                  name='%s%s_conv_proj' % (name, suffix), ratio=ratio)
    return proj


def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix='', ratio=-1.):
    identity = data
    for i in range(num_block):
        if name=="res_2" and i==0 and ratio>=0.:
            shortcut = Linear(data, num_filter=num_out, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=name+'sc', suffix=suffix, ratio=ratio)
        else:
            shortcut = identity
        conv = DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group,
                         name='%s%s_block' % (name, suffix), suffix='%d' % i, ratio=ratio)
        # identity = conv + shortcut
        identity = symbol_octconv.ElementWiseSum(*[shortcut, conv], name=('%s%s_blob%d_sum' % (name, suffix, i)))
    return identity
	
def Residual2(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix='', ratio=-1.):
    identity = data
    for i in range(num_block):
        if name=="res_2" and i==0 and ratio>=0.:
            shortcut = Linear(data, num_filter=num_out, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=name+'sc', suffix=suffix, ratio=ratio)
        else:
            shortcut = identity
        conv = DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group,
                         name='%s%s_block' % (name, suffix), suffix='%d' % i, ratio=ratio)
        conv1 = DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group,
                         name='%s%s_block1' % (name, suffix), suffix='%d' % i, ratio=ratio)
        concat = symbol_octconv.Concat(conv, conv1, name=('%s%s_blob%d_concat' % (name, suffix, i)))
        conv_concat = ConvOnly(concat, num_filter=num_out, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=('%s%s_blob%d_concat_linear' % (name, suffix, i)), suffix=suffix, ratio=ratio)	
        # identity = conv + shortcut
        identity = symbol_octconv.ElementWiseSum(*[shortcut, conv_concat], name=('%s%s_blob%d_sum' % (name, suffix, i)))
    return identity


def get_symbol():
    num_classes = config.emb_size
    ratio = config.ratio
    print('in_network', config)
    fc_type = config.net_output
    data = mx.symbol.Variable(name="data")
    data = data - 127.5
    data = data * 0.0078125
    blocks = config.net_blocks
    conv_1 = Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1")
    if blocks[0] == 1:
        conv_2_dw = Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                         name="conv_2_dw", ratio=ratio)
    else:
        conv_2_dw = Residual(conv_1, num_block=blocks[0], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                             num_group=64, name="res_2", ratio=ratio)
    conv_23 = DResidual(conv_2_dw, num_out=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128, name="dconv_23",
                        ratio=ratio)
    conv_3 = Residual(conv_23, num_block=blocks[1], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128,
                      name="res_3", ratio=ratio)
    conv_34 = DResidual(conv_3, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, name="dconv_34", ratio=ratio)
    conv_4 = Residual(conv_34, num_block=blocks[2], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      num_group=256, name="res_4", ratio=ratio)
    conv_45 = DResidual(conv_4, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, name="dconv_45", ratio=0.)
    conv_5 = Residual(conv_45, num_block=blocks[3], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      num_group=256, name="res_5", ratio=0.)
    conv_6_sep = Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep", ratio=0.)

    fc1 = symbol_utils.get_fc1(conv_6_sep, num_classes, fc_type)

    # plot network architecture
    #digraph = mx.viz.plot_network(fc1, shape={'data': (1, 3, 112, 112)}, save_format='pdf',
    #                              node_attrs={"shape": "oval", "fixedsize": "false"})
    #digraph.render(filename='octconv_fmobilefacenet')

    return fc1

