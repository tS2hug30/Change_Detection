#! /usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
import os
import random
import math
import sys


import numpy as np
from glob import glob
import cv2
import random
import os
import chainer
from chainer import optimizers
from chainer import serializers
from chainer import cuda
from chainer import functions as F

class diff_loss:
    def cu_conv(self, feat_map):
        feat_map = F.transpose(feat_map,axes=(0, 2, 3, 1))
        return feat_map


    def min_max(self, x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return result

    # 学習用
    def batch_split(self, feat_map):

        grid = 4
        x_grid = feat_map.shape[1]/grid
        y_grid = feat_map.shape[2]/grid


        #高さを分割
        for i in range(x_grid):
            x_grid_feat = F.reshape(feat_map[:,i*grid:(i+1)*grid],(4,grid,512,64))
            x_dim_up = F.expand_dims(x_grid_feat,axis=2)
            if i == 0:
                x_feat = x_dim_up
            else:
                x_feat = F.concat((x_feat,x_dim_up),axis=2)
        #横を分割
        for i in range(y_grid):
            y_grid_feat = F.reshape(x_feat[:,:,:,i*grid:(i+1)*grid],(4,grid,x_grid,grid,64))
            dim_up = F.expand_dims(y_grid_feat,axis=4)
            if i == 0:
                grid_feat = dim_up
            else:
                grid_feat = F.concat((grid_feat,dim_up),axis=4)

        #次元入れ替え変更
        grid_feat = F.transpose(grid_feat,axes=(0,2,4,5,1,3))
        grid_feat = F.reshape(grid_feat,(4,x_grid*y_grid,64*grid*grid))

        #正規化とL2
        Norm_feat = F.normalize(grid_feat, eps=1e-05, axis=2)
        diff_feat1 = F.sqrt(F.sum(F.square(Norm_feat[0]-Norm_feat[2]),axis=1))
        diff_feat2 = F.sqrt(F.sum(F.square(Norm_feat[1]-Norm_feat[3]),axis=1))
        
        return diff_feat1,diff_feat2

    # テスト用
    def batch_split1(self, feat_map1, feat_map2):

        feat_map1 = feat_map1[0]
        feat_map2 = feat_map2[0]
        grid = 4
        x_grid = feat_map1.shape[0]/grid
        y_grid = feat_map1.shape[1]/grid

        #特徴マップ合体
        feat_map1 = F.expand_dims(feat_map1,axis=0)
        feat_map2 = F.expand_dims(feat_map2,axis=0)
        feat_map = F.concat((feat_map1,feat_map2),axis=0)

        #高さを分割
        for i in range(x_grid):
            x_grid_feat = F.reshape(feat_map[:,i*grid:(i+1)*grid],(2,grid,512,64))
            x_dim_up = F.expand_dims(x_grid_feat,axis=2)
            if i == 0:
                x_feat = x_dim_up
            else:
                x_feat = F.concat((x_feat,x_dim_up),axis=2)
        #横を分割
        for i in range(y_grid):
            y_grid_feat = F.reshape(x_feat[:,:,:,i*grid:(i+1)*grid],(2,grid,x_grid,grid,64))
            dim_up = F.expand_dims(y_grid_feat,axis=4)
            if i == 0:
                grid_feat = dim_up
            else:
                grid_feat = F.concat((grid_feat,dim_up),axis=4)

        #次元入れ替え変更
        grid_feat = F.transpose(grid_feat,axes=(0,2,4,5,1,3))
        grid_feat = F.reshape(grid_feat,(2,x_grid,y_grid,64*grid*grid))
        Norm_feat = F.normalize(grid_feat, eps=1e-05, axis=3)
        diff_feat = F.sqrt(F.sum(F.square(Norm_feat[0]-Norm_feat[1]),axis=2))
        return diff_feat


    def match(self, diff, gt):
        gt_flat = F.reshape(gt,(gt.shape[0]*gt.shape[1],))
        gt_diff = F.sqrt(F.sum(F.square(diff-gt_flat),axis=0))
        if math.isnan(cuda.to_cpu(gt_diff.data)):
            sys.exit()
        return gt_diff
               
