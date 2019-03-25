#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import os
import random
import cv2
from glob import glob
import math
from PIL import Image
#os.environ["CHAINER_TYPE_CHECK"] = "0"

import chainer
from chainer import optimizers
from chainer import serializers
from chainer import cuda
import gc

from unet import Model
from dataloader import Dataloader
from diff_loss import diff_loss

'''-----------------------------------------------------'''
# 学習用の入力画像が入ったフォルダのPATH
IMAGE_PATH1 = "../Image_all/Original/t0/*.jpg"
IMAGE_PATH2 = "../Image_all/Original/t1/*.jpg"
# 学習用の教師画像が入ったフォルダのPATH
LABEL_PATH1 = "../Image_all/Gray_ano/t0/*.bmp"
LABEL_PATH2 = "../Image_all/Gray_ano/t1/*.bmp"
# 学習時のバッチサイズ(偶数のみ)
BATCH_SIZE = 4

# ネットワークへの入力画像サイズ 32で割り切れないとダメっぽい
HEIGHT = 224
WIDTH = 1024

# 使用するGPU番号
GPU_ID = 0

# Data augmentationの有無
DA = True
'''-----------------------------------------------------'''


def train():
    # segnet.pyのネットワークの呼び出し
    model = Model(train=True)
    #serializers.load_npz('./TRAIN/'+ '96000.model', model)
    cuda.get_device(GPU_ID).use()
    model.to_gpu()

    # 最適化設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    #serializers.load_npz('./TRAIN/' + '96000.state', optimizer)


    load = Dataloader()
    Diff_loss = diff_loss()


    image = np.zeros([BATCH_SIZE, 3, HEIGHT, WIDTH], dtype=np.float32)
    label = np.zeros([BATCH_SIZE, HEIGHT, WIDTH], dtype=np.int32)

    TIME = 0.0

    for iteration in xrange(100001):# 300001 → 50001
        start = time.time()
        #iteration = iteration + 96000
        # 誤差とoptimizerの初期化
        loss = 0.0
        #optimizer.zero_grads()
        model.cleargrads()

        # 入力画像と教師画像の読み込み
        image,label,gt_truth=load.Train(IMAGE_PATH1,IMAGE_PATH2,LABEL_PATH1,LABEL_PATH2,BATCH_SIZE,HEIGHT,WIDTH,DA,iteration)
        label = label / 20 # 画素値がclass0：0，class1：20，class2：40 …となっているため

        image_gpu = cuda.to_gpu(image)
        INPUT = chainer.Variable(image_gpu)
        label_gpu = cuda.to_gpu(label)
        ANSER = chainer.Variable(label_gpu)
        truth_gpu = cuda.to_gpu(gt_truth)
        TRUTH = chainer.Variable(truth_gpu)
        

        # 誤差の算出
        loss1,Dif_feat = model(INPUT, ANSER)
        Dif_feat = Diff_loss.cu_conv(Dif_feat)
        diff1,diff2 =Diff_loss.batch_split(Dif_feat)
        loss2 = Diff_loss.match(diff1,TRUTH[0])
        loss3 = Diff_loss.match(diff2,TRUTH[1])
        print("0.7",loss1)
        loss = 0.7*loss1 + 0.3*(loss2+loss3)
        print loss.data

        # ネットワークの最適化
        loss.backward()
        optimizer.update()
        gc.collect()

        # lossの表示
        end = time.time()
        TIME += end - start
        if iteration % 1 == 0:
            f = open('./TRAIN/loss.txt', 'a') # textデータに出力・保存
            print >> f, ("iteration:{}, loss:{}, time:{}".format(iteration, loss.data, TIME))
            f.close()
            TIME = 0

        # 学習データの保存
        if (iteration % 1000 == 0)  and (iteration != 0):
            model.to_cpu() # (テスト時のGPU依存問題を防ぐため一旦CPUに移動)
            serializers.save_npz('./TRAIN/' + str(iteration) + '.model', model)
            serializers.save_npz('./TRAIN/' + str(iteration) + '.state', optimizer)
            model.to_gpu()

        # iteration回数の表示
        print ([str( iteration + 1) + "iteration"])


if __name__ == '__main__':
    train()
