import sys
import os
import mxnet as mx
import symbol_utils
import mxnet.ndarray as nd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config


def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)
    act = Act(data=bn, act_type=config.net_act, name='%s%s_relu' %(name, suffix))
    return act
    
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)    
    return bn

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv    

    
def DResidual(data, num_out=1, num_prune=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_prune, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_prune, num_group=num_prune, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj
  
def DResidual2(data, num_out=1, num_prune=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_prune, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_prune, num_group=num_prune, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_prune, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    conv_dw1 = Conv(data=proj, num_filter=num_prune, num_group=num_prune, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw_1' %(name, suffix))
    proj1 = Linear(data=conv_dw1, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj_1' %(name, suffix))
    return proj1
  
def Residual2(data, num_block=1, num_out=1, num_prune=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv1=DResidual(data=identity, num_out=num_out, num_prune=num_prune, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	conv2=DResidual2(data=identity, num_out=num_out, num_prune=num_prune, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block_' %(name, suffix), suffix='%d'%i)
    	combine = mx.symbol.Concat(conv1, conv2, dim =1 )
    	conv = mx.sym.Convolution(data=combine, num_filter=num_out, kernel=(1,1), stride=(1,1), pad=(0, 0), name='%s_conv_%d' %(name, i))
    	conv = conv + shortcut
    	identity=conv
    return identity
  
def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	identity=conv+shortcut
    return identity
        
def get_symbol():
    num_classes = config.emb_size
    print('in_network', config)
    fc_type = config.net_output
    data = mx.symbol.Variable(name="data")
    data = data-127.5
    data = data*0.0078125
    blocks = config.net_blocks
    data_shape = {'data':(1,3,112,112)}
    conv_1 = Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1")
    list_param=[]
    list_name=[]
    
#    if blocks[0]==1:
#      conv_2_dw = Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw")
#    else:
    conv_2_dw = Residual2(conv_1, num_block=blocks[0], num_out=64, num_prune=32, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=64, name="res_2")
#    list_param=conv_2_dw.infer_shape(data=(1,3,112,112))[0][1:]
#    list_name=conv_2_dw.list_arguments()[1:]
#    i=0
#    for i in range(len(list_param)):
#        print('i: {}, list_name: {}, list_param: {}'.format(i, list_name[i], list_param[i]))
#        i=i+1
    conv_23 = DResidual(conv_2_dw, num_out=64, num_prune=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128, name="dconv_23")

#    list_param=conv_23.infer_shape(data=(1,3,112,112))[0][1:]
#    list_name=conv_23.list_arguments()[1:]
#    i=0
#    for i in range(len(list_param)):
#        print('i: {}, list_name: {}, list_param: {}'.format(i, list_name[i], list_param[i]))
#        i=i+1

    conv_3 = Residual2(conv_23, num_block=blocks[1], num_out=64, num_prune=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128, name="res_3")

#    list_param=conv_3.infer_shape(data=(1,3,112,112))[0][1:]
#    list_name=conv_3.list_arguments()[1:]
#    i=0
#    for i in range(len(list_param)):
#        print('i: {}, list_name: {}, list_param: {}'.format(i, list_name[i], list_param[i]))
#        i=i+1
    
    conv_34 = DResidual(conv_3, num_out=128, num_prune=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, name="dconv_34")
    conv_4 = Residual2(conv_34, num_block=blocks[2], num_out=128, num_prune=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_4")
    conv_45 = DResidual(conv_4, num_out=128, num_prune=256, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, name="dconv_45")
    conv_5 = Residual2(conv_45, num_block=blocks[3], num_out=128, num_prune=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_5")
    conv_6_sep = Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep")
    
    fc1 = symbol_utils.get_fc1(conv_6_sep, num_classes, fc_type)
#    list_param=fc1.infer_shape(data=(1,3,112,112))[0][1:]
#    list_name=fc1.list_arguments()[1:]
#    i=0
#    k=0
#    for i in range(len(list_param)):
#        #if len(list_param[i])==4:
#        print('k: {}, list_name: {}, list_param: {}'.format(k, list_name[i], list_param[i]))
#        k=k+1
#        i=i+1
#    
    return fc1

#get_symbol()