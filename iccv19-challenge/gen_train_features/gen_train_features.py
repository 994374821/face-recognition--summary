from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from datetime import datetime
import os.path
from easydict import EasyDict as edict
import time
import json
import sys
import numpy as np
import importlib
import itertools
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
import mxnet as mx
from mxnet import ndarray as nd
import numpy as np

# sys.path.append("/home/gaomingda/insightface/recognition")
# from image_iter import FaceImageIter

# from image_iter_gen_feature import FaceImageIter

image_shape = None
net = None
data_size = 1862120
emb_size = 0
use_flip = True



def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_feature(input_blob, batch_size):
  global emb_size

  data = mx.nd.array(input_blob)
  db = mx.io.DataBatch(data=(data,))
  net.model.forward(db, is_train=False)
  _embedding = net.model.get_outputs()[0].asnumpy()
  if emb_size==0:
    emb_size = _embedding.shape[1]
    print('set emb_size to ', emb_size)
  embedding = np.zeros((batch_size, emb_size), dtype=np.float32)
  embedding = _embedding
  # embedding = sklearn.preprocessing.normalize(embedding)
  return embedding



def write_bin(path, m):
  rows, cols = m.shape
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', rows,cols,cols*4,5))
    f.write(m.data)

def main(args):
  # sys.path.append("/home/gaomingda/insightface/recognition")
  from image_iter import FaceImageIter

  global image_shape
  global net

  print(args)
  ctx = []
  cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
  if len(cvd)>0:
    for i in xrange(len(cvd.split(','))):
      ctx.append(mx.gpu(i))
  if len(ctx)==0:
    ctx = [mx.cpu()]
    print('use cpu')
  else:
    print('gpu num:', len(ctx))
  image_shape = [int(x) for x in args.image_size.split(',')]
  vec = args.model.split(',')
  assert len(vec)>1
  prefix = vec[0]
  epoch = int(vec[1])
  print('loading',prefix, epoch)
  net = edict()
  net.ctx = ctx
  net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
  #net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
  all_layers = net.sym.get_internals()
  net.sym = all_layers['fc1_output']
  net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names = None)
  net.model.bind(data_shapes=[('data', (args.batch_size, 3, image_shape[1], image_shape[2]))])
  net.model.set_params(net.arg_params, net.aux_params)


  train_dataiter = FaceImageIter(
    batch_size=4,
    data_shape=(3, 112, 112),
    path_imgrec=args.input_data,
    shuffle=True,
    rand_mirror=False,
    mean=None,
    cutoff=False,
    color_jittering=0,
    images_filter=0,
  )
  data_size = train_dataiter.num_samples()
  i = 0
  fstart = 0

  features_all = np.zeros((data_size, 512), dtype=np.float32)
  features_all_flip = np.zeros((data_size, 512), dtype=np.float32)

  # features_all = np.zeros((102, 512), dtype=np.float32)
  # features_all_flip = np.zeros((102, 512), dtype=np.float32)

  data_buff = nd.empty((args.batch_size, 3, 112, 112))
  count = 0
  for i in range(train_dataiter.num_samples()):
    if i%1000==0:
      print("processing ",i)
    label, s, box, landmark = train_dataiter.next_sample()
    img = train_dataiter.imdecode(s)
    img = nd.transpose(img, axes=(2, 0, 1))
    data_buff[count] = img
    count += 1
    if count==args.batch_size:
      embedding = get_feature(data_buff, args.batch_size)
      count = 0
      fend = fstart+embedding.shape[0]

      #print('writing', fstart, fend)
      features_all[fstart:fend,:] = embedding
      # flipped image
      data_buff_flip = mx.ndarray.flip(data=data_buff, axis=3)
      embedding_fliped = get_feature(data_buff_flip, args.batch_size)
      features_all_flip[fstart:fend, :] = embedding_fliped

      fstart = fend

    # if i==102:
    #   break

  if count>0:
    embedding = get_feature(data_buff, args.batch_size)
    fend = fstart+count
    print('writing', fstart, fend)
    features_all[fstart:fend,:] = embedding[:count, :]

    # flipped image
    data_buff_flip = mx.ndarray.flip(data=data_buff, axis=3)
    embedding_fliped = get_feature(data_buff_flip, args.batch_size)
    features_all_flip[fstart:fend, :] = embedding_fliped[:count, :]

  # write_bin(args.output, features_all)
  #os.system("bypy upload %s"%args.output)
  print("save features ...")
  features_all.tofile('train_features_oct200')



  print("save train_features_flip ...")
  features_all_flip.tofile('train_features_flip_oct200')

  # np.savetxt('train_features', features_all)
  # np.savetxt('train_features_flip', features_all_flip)


def check_features(args):
  from image_iter_gen_feature import FaceImageIter
  global image_shape
  global net

  print(args)
  ctx = []
  cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
  if len(cvd) > 0:
    for i in xrange(len(cvd.split(','))):
      ctx.append(mx.gpu(i))
  if len(ctx) == 0:
    ctx = [mx.cpu()]
    print('use cpu')
  else:
    print('gpu num:', len(ctx))
  image_shape = [int(x) for x in args.image_size.split(',')]
  vec = args.model.split(',')
  assert len(vec) > 1
  prefix = vec[0]
  epoch = int(vec[1])
  print('loading', prefix, epoch)
  net = edict()
  net.ctx = ctx
  net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
  # net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
  all_layers = net.sym.get_internals()
  net.sym = all_layers['fc1_output']
  net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
  net.model.bind(data_shapes=[('data', (args.batch_size, 3, image_shape[1], image_shape[2]))])
  net.model.set_params(net.arg_params, net.aux_params)

  train_dataiter = FaceImageIter(
    batch_size=args.batch_size,
    data_shape=(3, 112, 112),
    path_imgrec=args.input_data,
    shuffle=True,
    rand_mirror=False,
    mean=None,
    cutoff=False,
    color_jittering=0,
    images_filter=0,
  )

  for i in range(10):
    db, features_data = train_dataiter.next()
    net.model.forward(db, is_train=False)
    embedding = net.model.get_outputs()[0].asnumpy()

    print((embedding==features_data).any())


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', type=int, help='', default=64)
  parser.add_argument('--image_size', type=str, help='', default='3,112,112')
  parser.add_argument('--input_data', type=str, help='', default="/home/gaomingda/insightface/datasets/ms1m-retinaface-t1/train.rec")
  parser.add_argument('--output', type=str, help='', default='')
  parser.add_argument('--model', type=str, help='', default='/home/gaomingda/insightface/recognition/models/octconv-arcface-retina/model,1')
  parser.add_argument('--debug', action='store_true', default=False)
  return parser.parse_args(argv)

if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  print(args.debug)
  if not args.debug:
    main(args)
  else:
    check_features(args)


