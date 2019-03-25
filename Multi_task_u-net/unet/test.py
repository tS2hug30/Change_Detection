#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import cv2
from glob import glob
import math
from PIL import Image

import chainer
from chainer import optimizers
from chainer import serializers
from chainer import cuda

from unet import Model
#from fcn import Model

from dataloader import Dataloader
from diff_loss import diff_loss

'''-----------------------------------------------------'''
# テスト用の入力画像が入ったフォルダのPATH
RGB_PATH0 = "../Image_all/Test/t0/ORIGINAL/*.jpg"
RGB_PATH1 = "../Image_all/Test/t1/ORIGINAL/*.jpg"
# テスト用の教師画像が入ったフォルダのPATH
LABEL_PATH = "../Image_all/Test/t0/ORIGINAL/*.jpg"

# ネットワークへの入力画像サイズ
# 国際航業データセット時
WIDTH = 1024
HEIGHT = 512
# PCDデータセット時
#WIDTH = 1024
#HEIGHT = 224

# 学習モデルのPATH
MODEL_PATH = "./TRAIN/"
# 使用するGPU番号
GPU_ID = 0

'''-----------------------------------------------------'''

#0~1正規化
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

#1になるように正規化
def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2

#特徴マップを正規化
def feat_normalize(feat):
    feat.to_cpu()
    feat = feat.data[0,:,:,:]
    feat = feat.transpose(1, 2, 0)
    h,w,c = feat.shape
    return h,w, feat

#可視化
def Visual_feat(Visual,num,pool_num):
    Visual.to_cpu()
    feat = Visual.data[0,:,:,:]
    feat_mean = feat[0]
    c,h,w = feat.shape
    for kazu in range(c):
        feat_mean += feat[kazu]
        feat_pieces = min_max(feat[kazu]) * 255
    feat_mean = min_max(feat_mean) * 255
    feat_mean = cv2.resize(feat_mean,(1024,224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("./feat_visual/"+str(num)+str(pool_num)+"mean.png", feat_mean)

def test(label_colours):
    args = sys.argv
    if (len(args) != 3) or (args[1].count('.model') != 1) or (args[2].isdigit() != True):
        if len(args) != 3:
            print("Argument : "+str(len(args)-1)+" != 2")
            print("~~Python [program] [model] [threshold]~~")
        if args[1].count('.model') != 1:
            print("Argument1 : Please write the model name.")
        if args[2].isdigit() != True:
            print("Argument2 : Please write the threshold value.")
        sys.exit(1)
    tutui = int(args[2])

    # net.pyのネットワークの呼び出し
    model = Model(train=False)
    cuda.get_device(GPU_ID).use()
    model.to_gpu()

    # 最適化設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # 学習モデルの読み込み
    serializers.load_npz(MODEL_PATH+args[1], model)

    load = Dataloader()
    Diff_loss = diff_loss()
    TIMER = 0.0

    sample = glob(RGB_PATH0)
    sample_num = len(sample)
    for num in range(sample_num):
        image0, label, filename = load.Test(RGB_PATH0, LABEL_PATH, num, HEIGHT, WIDTH)
        image1, label, filename = load.Test(RGB_PATH1, LABEL_PATH, num, HEIGHT, WIDTH)

        image_gpu0 = cuda.to_gpu(image0)
        image_gpu1 = cuda.to_gpu(image1)
        with chainer.using_config('enable_backprop', False):
            INPUT0 = chainer.Variable(image_gpu0)
        with chainer.using_config('enable_backprop', False):
            INPUT1 = chainer.Variable(image_gpu1)
        label_gpu = cuda.to_gpu(label)
        with chainer.using_config('enable_backprop', False):
            ANSER = chainer.Variable(label_gpu)

        START = time.clock()
        pred0,Visual5_0 = model(INPUT0, ANSER)
        pred1,Visual5_1 = model(INPUT1, ANSER)
        END = time.clock()

        #相違度計算用
        Dif_feat0 = Diff_loss.cu_conv(Visual5_0)
        Dif_feat1 = Diff_loss.cu_conv(Visual5_1)
        diff = Diff_loss.batch_split1(Dif_feat0,Dif_feat1)
        h, w = diff.shape
        diff.to_cpu()
        diff = diff.data

        segmented_image1 = np.zeros((h, w, 1), dtype=np.float32)
        segmented_image2 = np.zeros((h, w, 1), dtype=np.float32)
        segmented_image3 = np.zeros((h, w, 1), dtype=np.float32)

        segmented_image1 = min_max(diff) * 255
        for i in range(len(segmented_image1)):
            for j in range(len(segmented_image1[0])):
                if segmented_image1[i][j] <= tutui:
                    segmented_image2[i][j][0] = 0
                else:
                    segmented_image2[i][j][0] = segmented_image1[i][j]
                if segmented_image1[i][j] <= tutui:
                    segmented_image3[i][j][0] = 255
                else:
                    segmented_image3[i][j][0] = 0
        """---------------------------------------------------------------"""
        segmented_image2 = np.array(segmented_image2).astype('uint8')
        segmented_image2 = cv2.applyColorMap(segmented_image2, cv2.COLORMAP_JET)
        """---------------------------------------------------------------"""
        # 閾値処理後のカラーマップ
        segmented_image2 = cv2.resize(segmented_image2,(WIDTH,HEIGHT), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("./feat_visual/threshold/"+str(num)+".png", segmented_image2)
        # 上のやつを2値化
        segmented_image3 = cv2.resize(segmented_image3,(WIDTH,HEIGHT), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("./feat_visual/"+str(tutui)+"/"+str(num)+".png", segmented_image3)
        # 通常のバージョン
        segmented_image1 = np.array(segmented_image1).astype('uint8')
        img_raw1 = cv2.applyColorMap(segmented_image1, cv2.COLORMAP_JET)
        img_raw1 = cv2.resize(img_raw1,(WIDTH,HEIGHT), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("./feat_visual/"+str(num)+".png", img_raw1)

        # 特徴マップ可視化用
        #Visual_feat(Visual5_0,num,1)
        #Visual_feat(Visual5_1,num,2)

        pred0.to_cpu()
        pred1.to_cpu()
        result0 = np.squeeze(pred0.data[0,:,:,:])
        result1 = np.squeeze(pred1.data[0,:,:,:])
        ind0 = np.argmax(result0, axis=0)
        ind1 = np.argmax(result1, axis=0)

        rgb ,rgb_gt = segmentation(ind0,label,label_colours)

        # Segmentation(before)
        cv2.imwrite("./OUTPUT/result0/" + filename + ".png", rgb)
        cv2.imwrite("./OUTPUT/true/" + filename + ".png", rgb_gt)
        image = np.squeeze(image0[0,:,:,:])
        image = (image.swapaxes(0, 2)).swapaxes(0, 1)
        cv2.imwrite("./OUTPUT/image0/" + filename + ".png", image)

        rgb ,rgb_gt = segmentation(ind1,label,label_colours)

        # Segmentation(after)
        cv2.imwrite("./OUTPUT/result1/" + filename + ".png", rgb)
        image = np.squeeze(image1[0,:,:,:])
        image = (image.swapaxes(0, 2)).swapaxes(0, 1)
        cv2.imwrite("./OUTPUT/image1/" + filename + ".png", image)


        print(str(num+1) + "image was saved.")
        TIMER += END - START

    print(TIMER)


def color_array():
    color = np.array([
           [ 70,  70,  70],
           [ 70, 130, 180],
           [243,  35, 232]])
    color = color[:, ::-1]
    return color

def segmentation(ind,label,label_colours):
    r = ind.copy()
    g = ind.copy()
    b = ind.copy()

    r_gt = label.copy()
    g_gt = label.copy()
    b_gt = label.copy()

    for l in range(0, label_colours.shape[0]):
        r[ind==l] = label_colours[l,0]
        g[ind==l] = label_colours[l,1]
        b[ind==l] = label_colours[l,2]
        r_gt[label==l] = label_colours[l,0]
        g_gt[label==l] = label_colours[l,1]
        b_gt[label==l] = label_colours[l,2]

    rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b
    rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb_gt[:,:,0] = r_gt
    rgb_gt[:,:,1] = g_gt
    rgb_gt[:,:,2] = b_gt

    return rgb,rgb_gt

if __name__ == '__main__':
    color = color_array()
    test(color)
