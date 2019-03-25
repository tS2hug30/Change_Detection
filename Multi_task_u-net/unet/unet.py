#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.pooling import max_pooling_2d


DEBUG = False

class Model(chainer.Chain):
    def __init__(self, train):
        self.train = train

        super(Model, self).__init__(
            Input_bn=L.BatchNormalization(3),

            Econv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            Ebn1_1=L.BatchNormalization(64),
            Econv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            Ebn1_2=L.BatchNormalization(64),

            Econv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            Ebn2_1=L.BatchNormalization(128),
            Econv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            Ebn2_2=L.BatchNormalization(128),

            Econv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            Ebn3_1=L.BatchNormalization(256),
            Econv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            Ebn3_2=L.BatchNormalization(256),
            Econv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            Ebn3_3=L.BatchNormalization(256),

            Econv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            Ebn4_1=L.BatchNormalization(512),
            Econv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            Ebn4_2=L.BatchNormalization(512),
            Econv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            Ebn4_3=L.BatchNormalization(512),

            Econv5_1=L.Convolution2D(512, 1024, 3, stride=1, pad=1),
            Ebn5_1=L.BatchNormalization(1024),
            Econv5_2=L.Convolution2D(1024, 1024, 3, stride=1, pad=1),
            Ebn5_2=L.BatchNormalization(1024),
            Econv5_3=L.Convolution2D(1024, 1024, 3, stride=1, pad=1),
            Ebn5_3=L.BatchNormalization(1024),


            Dconv5_3=L.Convolution2D(1024, 1024, 3, stride=1, pad=1),
            Dbn5_3=L.BatchNormalization(1024),
            Dconv5_2=L.Convolution2D(1024, 1024, 3, stride=1, pad=1),
            Dbn5_2=L.BatchNormalization(1024),
            Dconv5_1=L.Convolution2D(1024, 512, 3, stride=1, pad=1),
            Dbn5_1=L.BatchNormalization(512),

#            Dconv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            Dconv4_3=L.Convolution2D(1024, 512, 3, stride=1, pad=1),
            Dbn4_3=L.BatchNormalization(512),
            Dconv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            Dbn4_2=L.BatchNormalization(512),
            Dconv4_1=L.Convolution2D(512, 256, 3, stride=1, pad=1),
            Dbn4_1=L.BatchNormalization(256),

#            Dconv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            Dconv3_3=L.Convolution2D(512, 256, 3, stride=1, pad=1),
            Dbn3_3=L.BatchNormalization(256),
            Dconv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            Dbn3_2=L.BatchNormalization(256),
            Dconv3_1=L.Convolution2D(256, 128, 3, stride=1, pad=1),
            Dbn3_1=L.BatchNormalization(128),

#            Dconv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            Dconv2_2=L.Convolution2D(256, 128, 3, stride=1, pad=1),
            Dbn2_2=L.BatchNormalization(128),
            Dconv2_1=L.Convolution2D(128, 64, 3, stride=1, pad=1),
            Dbn2_1=L.BatchNormalization(64),

#            Dconv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            Dconv1_2=L.Convolution2D(128, 64, 3, stride=1, pad=1),
            Dbn1_2=L.BatchNormalization(64),
            Dconv1_1=L.Convolution2D(64, 3, 3, stride=1, pad=1), #最終的なクラス数を指定
        )


    def Encoder(self, x):
        y = self.Input_bn(x)

        y = F.relu(self.Ebn1_1(self.Econv1_1(y)))
        if DEBUG: print("conv1-1")
        if DEBUG: print(y.shape)
        y4 = F.relu(self.Ebn1_2(self.Econv1_2(y)))
        if DEBUG: print("conv1-2")
        if DEBUG: print(y4.shape)
        y,indexes4 = F.max_pooling_2d(y4, ksize=2, stride=2, return_indices=True)
        if DEBUG: print("maxpool1")
        if DEBUG: print(y.shape)
        if DEBUG: print(self.max_pooling1.indexes.shape)

        y = F.relu(self.Ebn2_1(self.Econv2_1(y)))
        if DEBUG: print("conv2-1")
        if DEBUG: print(y.shape)
        y3 = F.relu(self.Ebn2_2(self.Econv2_2(y)))
        if DEBUG: print("conv2-2")
        if DEBUG: print(y3.shape)
        y,indexes3 = F.max_pooling_2d(y3, ksize=2, stride=2, return_indices=True)
        if DEBUG: print("maxpool2")
        if DEBUG: print(y.shape)
        if DEBUG: print(self.max_pooling2.indexes.shape)

        y = F.relu(self.Ebn3_1(self.Econv3_1(y)))
        if DEBUG: print("conv3-1")
        if DEBUG: print(y.shape)
        y = F.relu(self.Ebn3_2(self.Econv3_2(y)))
        if DEBUG: print("conv3-2")
        if DEBUG: print(y.shape)
        y2 = F.relu(self.Ebn3_3(self.Econv3_3(y)))
        if DEBUG: print("conv3-3")
        if DEBUG: print(y2.shape)
        y,indexes2 = F.max_pooling_2d(y2, ksize=2, stride=2, return_indices=True)
        if DEBUG: print("maxpool3")
        if DEBUG: print(y.shape)
        if DEBUG: print(self.max_pooling3.indexes.shape)
        
        y = F.relu(self.Ebn4_1(self.Econv4_1(y)))
        y = F.relu(self.Ebn4_2(self.Econv4_2(y)))
        y1 = F.relu(self.Ebn4_3(self.Econv4_3(y)))
        y,indexes1 = F.max_pooling_2d(y1, ksize=2, stride=2, return_indices=True)
        if DEBUG: print("maxpool4")
        if DEBUG: print(y.shape)
        if DEBUG: print(self.max_pooling4.indexes.shape)

        y = F.relu(self.Ebn5_1(self.Econv5_1(y)))
        y = F.relu(self.Ebn5_2(self.Econv5_2(y)))
        y = F.relu(self.Ebn5_3(self.Econv5_3(y)))
        y,indexes = F.max_pooling_2d(y, ksize=2, stride=2, return_indices=True)
        if DEBUG: print("maxpool5")
        if DEBUG: print(y.shape)
        if DEBUG: print(self.max_pooling5.indexes.shape)

        return y,y1,y2,y3,y4,indexes,indexes1,indexes2,indexes3,indexes4


    def Decoder1(self,y,y1,y2,y3,y4,indexes,indexes1,indexes2,indexes3,indexes4):
        y = F.upsampling_2d(y, indexes, ksize=2, stride=2, cover_all=False)
        y = F.relu(self.Dbn5_3(self.Dconv5_3(y)))
        y = F.relu(self.Dbn5_2(self.Dconv5_2(y)))
        y = F.relu(self.Dbn5_1(self.Dconv5_1(y)))

        y = F.upsampling_2d(y, indexes1, ksize=2, stride=2, cover_all=False)
        Upsam2 = y
        y = F.concat((y, y1), axis=1) # U-Net:Encoderの出力をDecoderの入力に連結
        y = F.relu(self.Dbn4_3(self.Dconv4_3(y)))
        y = F.relu(self.Dbn4_2(self.Dconv4_2(y)))
        y = F.relu(self.Dbn4_1(self.Dconv4_1(y)))
        feat_Dconv4_1 = y

        y = F.upsampling_2d(y, indexes2, ksize=2, stride=2, cover_all=False)
        Upsam3 = y
        y = F.concat((y, y2), axis=1) # U-Net:Encoderの出力をDecoderの入力に連結
        y = F.relu(self.Dbn3_3(self.Dconv3_3(y)))
        y = F.relu(self.Dbn3_2(self.Dconv3_2(y)))
        y = F.relu(self.Dbn3_1(self.Dconv3_1(y)))
        feat_Dconv3_1 = y

        y = F.upsampling_2d(y, indexes3, ksize=2, stride=2, cover_all=False)
        Upsam4 = y
        y = F.concat((y, y3), axis=1) # U-Net:Encoderの出力をDecoderの入力に連結
        y = F.relu(self.Dbn2_2(self.Dconv2_2(y)))
        y = F.relu(self.Dbn2_1(self.Dconv2_1(y)))
        feat_Dconv2_1 = y

        y = F.upsampling_2d(y, indexes4, ksize=2, stride=2, cover_all=False)
        Upsam5 = y
        y = F.concat((y, y4), axis=1) # U-Net:Encoderの出力をDecoderの入力に連結
        y = F.relu(self.Dbn1_2(self.Dconv1_2(y)))
        y = self.Dconv1_1(y)
        feat_Dconv1_1 = y

        return y

    def Decoder2(self,y,y1,y2,y3,y4,indexes,indexes1,indexes2,indexes3,indexes4):
        y = F.upsampling_2d(y, indexes, ksize=2, stride=2, cover_all=False)
        y = F.relu(self.Dbn5_3(self.Dconv5_3(y)))
        y = F.relu(self.Dbn5_2(self.Dconv5_2(y)))
        y = F.relu(self.Dbn5_1(self.Dconv5_1(y)))

        y = F.upsampling_2d(y, indexes1, ksize=2, stride=2, cover_all=False)
        Upsam2 = y
        y = F.concat((y, y1), axis=1) # U-Net:Encoderの出力をDecoderの入力に連結
        y = F.relu(self.Dbn4_3(self.Dconv4_3(y)))
        y = F.relu(self.Dbn4_2(self.Dconv4_2(y)))
        y = F.relu(self.Dbn4_1(self.Dconv4_1(y)))
        feat_Dconv4_1 = y

        y = F.upsampling_2d(y, indexes2, ksize=2, stride=2, cover_all=False)
        Upsam3 = y
        y = F.concat((y, y2), axis=1) # U-Net:Encoderの出力をDecoderの入力に連結
        y = F.relu(self.Dbn3_3(self.Dconv3_3(y)))
        y = F.relu(self.Dbn3_2(self.Dconv3_2(y)))
        y = F.relu(self.Dbn3_1(self.Dconv3_1(y)))
        feat_Dconv3_1 = y

        y = F.upsampling_2d(y, indexes3, ksize=2, stride=2, cover_all=False)
        Upsam4 = y
        y = F.concat((y, y3), axis=1) # U-Net:Encoderの出力をDecoderの入力に連結
        y = F.relu(self.Dbn2_2(self.Dconv2_2(y)))
        y = F.relu(self.Dbn2_1(self.Dconv2_1(y)))
        feat_Dconv2_1 = y

        return y


    def __call__(self, x, t):
        # Encoder
        y,y1,y2,y3,y4,indexes,indexes1,indexes2,indexes3,indexes4 = self.Encoder(x)

        # Decoder
        Seg_feat = self.Decoder1(y,y1,y2,y3,y4,indexes,indexes1,indexes2,indexes3,indexes4)
        Dif_feat = self.Decoder2(y,y1,y2,y3,y4,indexes,indexes1,indexes2,indexes3,indexes4)

        if self.train == True:
            loss = F.softmax_cross_entropy(Seg_feat, t)
            return loss, Dif_feat

        elif self.train == False:
            pred = F.softmax(Seg_feat)
            return pred, Dif_feat
