#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from glob import glob
import cv2
import random
import os
from chainer import functions as F
from augmentation import Augmentation

GT_PATH = "../GT/Original/*.bmp"
grid = 8

class Dataloader:
    def Train(self, rgb_path1, rgb_path2, label_path1, label_path2, batch, height, width, da,iteration):
        if da: augmentation = Augmentation(height, width)

        IMAGE_FILE1 = glob(rgb_path1)
        IMAGE_FILE2 = glob(rgb_path2)
        LABEL_FILE1 = glob(label_path1)
        LABEL_FILE2 = glob(label_path2)
        TRUTH_FILE = glob(GT_PATH)

        IMAGE_FILE1.sort()
        IMAGE_FILE2.sort()
        LABEL_FILE1.sort()
        LABEL_FILE2.sort()
        TRUTH_FILE.sort()

        total_image = len(IMAGE_FILE1)
        size_image = (width, height)

        image = np.zeros((batch, height, width, 3)).astype(np.float32)
        label = np.zeros((batch, height, width)).astype(np.int32)
        truth = np.zeros((batch/2, height/grid, width/grid)).astype(np.float32)

        for counter in xrange(batch/2):
            choice = random.randint(0, total_image-1)

            readed_image1 = cv2.imread(IMAGE_FILE1[choice], 1)
            readed_image2 = cv2.imread(IMAGE_FILE2[choice], 1)
            readed_label1 = cv2.imread(LABEL_FILE1[choice], 0)
            readed_label2 = cv2.imread(LABEL_FILE2[choice], 0)
            readed_truth = cv2.imread(TRUTH_FILE[choice], 0)

            readed_image1 = cv2.resize(readed_image1, size_image, interpolation=cv2.INTER_LINEAR)
            readed_image2 = cv2.resize(readed_image2, size_image, interpolation=cv2.INTER_LINEAR)
            readed_label1 = cv2.resize(readed_label1, size_image, interpolation=cv2.INTER_NEAREST)
            readed_label2 = cv2.resize(readed_label2, size_image, interpolation=cv2.INTER_NEAREST)
            readed_truth = cv2.resize(readed_truth, size_image, interpolation=cv2.INTER_NEAREST)
            
            """--------------------トリミング--------------------"""
            h_start, h_end, w_start, w_end = 0, 0, 0, 0
            input_size = (width, height)

            size_rate = random.uniform(1, 2.7)
            cut_height = int(height / size_rate)
            cut_width = int(width / size_rate)

            while True:
                h_start = random.randint(0, height - cut_height)
                h_end = h_start + cut_height
                w_start = random.randint(0, width - cut_width)
                w_end = w_start + cut_width
                if h_start < int(height / 2) < h_end:
                    break

            readed_image1 = readed_image1[h_start:h_end, w_start:w_end]
            readed_image2 = readed_image2[h_start:h_end, w_start:w_end]
            readed_label1 = readed_label1[h_start:h_end, w_start:w_end]
            readed_label2 = readed_label2[h_start:h_end, w_start:w_end]
            readed_truth = readed_truth[h_start:h_end, w_start:w_end]

            readed_image1 = cv2.resize(readed_image1, size_image, interpolation=cv2.INTER_LINEAR)
            readed_image2 = cv2.resize(readed_image2, size_image, interpolation=cv2.INTER_LINEAR)
            readed_label1 = cv2.resize(readed_label1, size_image, interpolation=cv2.INTER_NEAREST)
            readed_label2 = cv2.resize(readed_label2, size_image, interpolation=cv2.INTER_NEAREST)
            readed_truth = cv2.resize(readed_truth, size_image, interpolation=cv2.INTER_NEAREST)
            """--------------------------------------------------------------------------------"""

            """--------------------DATA Augmentation--------------------"""
            method = -1
            reverse = random.randint(0, 1)
            if da: readed_image1, readed_label1, method = augmentation(readed_image1, readed_label1, method, reverse, 0)
            if da: readed_image2, readed_label2, method = augmentation(readed_image2, readed_label2, method, reverse, 0)
            if da: readed_image2, readed_truth, method = augmentation(readed_image2, readed_truth, method, reverse, 1)
            """--------------------------------------------------------------------------------"""
            """
            if counter == 0:
                cv2.imwrite("4.png", readed_image1)
                cv2.imwrite("5.png", readed_image2)
            else:
                cv2.imwrite("6.png", readed_image1)
                cv2.imwrite("7.png", readed_image2)
            """
            readed_image1 = cv2.resize(readed_image1, size_image, interpolation=cv2.INTER_LINEAR)
            readed_image2 = cv2.resize(readed_image2, size_image, interpolation=cv2.INTER_LINEAR)
            readed_label1 = cv2.resize(readed_label1, size_image, interpolation=cv2.INTER_NEAREST)
            readed_label2 = cv2.resize(readed_label2, size_image, interpolation=cv2.INTER_NEAREST)
            readed_truth = cv2.resize(readed_truth, size_image, interpolation=cv2.INTER_NEAREST)

            gt_image = np.zeros((height / grid, width / grid, 1), dtype=np.float32)
            for i in range(height / grid):
                for j in range(width / grid):
                    grids = readed_truth[i * grid:(i+1) * grid, j * grid:(j+1) * grid]
                    if np.sum(grids) >= (np.square(grid)*255)/2:
                        gt_image[i, j] = 0
                    else:
                        gt_image[i, j] = 1.41421356
            gt_image = cv2.resize(gt_image, (width/grid, height/grid), interpolation=cv2.INTER_NEAREST)


            image[counter, :, :, :] = readed_image1
            image[counter+batch/2, :, :, :] = readed_image2
            label[counter, :, :] = readed_label1
            label[counter+batch/2, :, :] = readed_label2
            truth[counter, :, :] = gt_image

        image = image.swapaxes(1, 3)
        image = image.swapaxes(2, 3)

        return image, label, truth


    def Test(self, rgb_path, label_path, num, height, width):
        IMAGE_FILE = glob(rgb_path)
        LABEL_FILE = glob(label_path)

        IMAGE_FILE.sort()
        LABEL_FILE.sort()

        size_image = (width, height)

        image = np.zeros((1, height, width, 3)).astype(np.float32)
        label = np.zeros((1, height, width)).astype(np.int32)

        img = cv2.imread(IMAGE_FILE[num], 1)
        resize = cv2.resize(img, size_image, interpolation=cv2.INTER_LINEAR)
        image[0, :, :, :] = resize

        img = cv2.imread(LABEL_FILE[num], 0)
        resize = cv2.resize(img, size_image, interpolation=cv2.INTER_NEAREST)
        label[0, :, :] = resize

        image = image.swapaxes(1, 3)
        image = image.swapaxes(2, 3)

        filename, ext = os.path.splitext(os.path.basename(IMAGE_FILE[num]))
        return image, label, filename
