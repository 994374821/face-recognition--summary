
# import mxnet as mx
# from config import config, default, generate_config
# import numpy as np
#
# data = mx.symbol.Variable(name="data")
#
# crop = mx.symbol.Crop(data, offset=(2, 1), h_w=(2, 2))
#
# a = np.random.rand(1, 3, 5, 5)
# print(a)
# ex = crop.bind(ctx=mx.cpu(), args={'data' : mx.nd.array(a)})
# out = ex.forward()
# print(out[0].asnumpy())

import numpy as np
import time

file_path = "/home/gaomingda/insightface/iccv19-challenge/test.txt"
file_path2 = "/home/gaomingda/insightface/iccv19-challenge/test2.txt"

# a = np.array(np.random.rand(10000, 512), dtype=np.float32)
# # print(a)
# np.savetxt('ttttttt', a)
# b = np.loadtxt('ttttttt')
# # print(b)
# print((a==b).all())
# print(1)

# a = np.array([[1,2], [3,5]], dtype=np.float32)
# a = np.zeros((5179510, 512), dtype=np.float32)
a = np.array(np.random.rand(1000,512), dtype=np.float32)

time1 = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(time1)

# with open(file_path2, 'wb') as f:
#     f.truncate()
#     f.write(a.data)
#
# with open(file_path, 'rb') as f:
#     lines = f.read()
time2 = time.strftime("%Y-%m-%d %H:%M:%S %Y",time.localtime())
print(time2)

a.tofile(file_path)

time3 = time.strftime("%Y-%m-%d %H:%M:%S %Y",time.localtime())
print(time3)

b = np.fromfile(file_path, dtype=np.float32)

time4 = time.strftime("%Y-%m-%d %H:%M:%S %Y",time.localtime())

# print(b)
b = b.reshape(1000, 512)
print((a==b).any())



print(time4)
