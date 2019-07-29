

### Lightweight Face Recognition Challenge(ICCV 2019) 心得

前段时间大概有一个半月的时间在参加Lightweight Face Recognition Challenge(ICCV 2019)的竞赛，虽然取得的成绩不理想，主要记录一下所做的工作与心得。

### Train

具体使用方法详见https://github.com/deepinsight/insightface

### OctConv
OctConv 能够降低模型的FLOPs,同时不会造成精度的损失。我们将普通卷积替换成OctConv，FLOPs计算结果如下：

| Method              | FLOPs   |
| --------------------| ------- | 
|y2(fmobilefacenet)   | 915.6M  | 
|octconv_mobilefacenet| 770.2M  |
| r100                | 22.5G   |
| octconv(r100)       | 12.6G   |

### 小模型采用知识蒸馏的策略
使用大模型的特征训练小模型
1、大模型特征提取
iccv19-challenge/gen_train_features/gen_train_features.py 
采用数组存储特征，数组下标即为图片id
2、训练
recognition文件夹下的train_embed.py

### 脏数据
对于脏数据参考论文《Noise-Tolerant Paradigm for Training Face Recognition CNNs》
code link：https://github.com/huangyangyu/NoiseFace
官方源码使用caffe实现，我们使用 MobulaOP 工具将其转换为mxnet版本，但是没有来得及训练
具体见MobulaOP/docs/tutorial文件夹，其中NoiseTolerant为实现的code， test_noisy_tolerant_op.py为测试脚本

同时也为caffe版本生产该人脸识别数据集的lmdb格式数据
详见recognition/createLmdb
1、运行gen_train_data.py生成图片与lable，保存在My_Files文件夹下
2、运行create_imagenet_lmdb.sh 生成lmdb

### 心得
大型数据库计算力很重要
多阅读文献，多试错，多交流
