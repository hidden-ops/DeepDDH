#!/usr/bin/python2.6
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk

# 计算DICE系数，即DSI
def calDSI(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s, DSI_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j] == 255:
                DSI_t += 1
    if DSI_t == 0:
        DSI_t = 1
    DSI = 2 * DSI_s / DSI_t
    # print(DSI)
    return DSI


# 计算VOE系数，即VOE
def calVOE(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    VOE_s, VOE_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                VOE_s += 1
            if binary_R[i][j] == 255:
                VOE_t += 1
    if (VOE_t + VOE_s) == 0:
        VOE_t = 1
    VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
    return VOE


# 计算RVD系数，即RVD
def calRVD(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    RVD_s, RVD_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                RVD_s += 1
            if binary_R[i][j] == 255:
                RVD_t += 1
    if RVD_s == 0:
        RVD_s = 1
    #print(RVD_t, RVD_s)
    RVD = RVD_t / RVD_s - 1
    return RVD


# 计算Prevision系数，即Precison
def calPrecision(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j] == 255:
                P_t += 1
    Precision = P_s / P_t
    return Precision


# 计算Recall系数，即Recall
def calRecall(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1
    if R_t == 0:
        Recall = 0.0001
    else:
        Recall = R_s / R_t
    return Recall

def calDice(binary_GT, binary_R):
    if np.max(binary_GT) == 255:
        pred = binary_GT / 255
    else:
        pred = binary_GT
    if np.max(binary_R) == 255:
        true = binary_R / 255
    else:
        true = binary_R
    union = true * pred
    dice = 2 * np.sum(union) / (np.sum(true) + np.sum(pred))
    return dice

def calHausdorff(binary_GT, binary_R):
    img_GT = cv2.imread(binary_GT, 0)
    img_R = cv2.imread(binary_R, 0)
    mrk = 1
    for i in range(img_GT.shape[0]):
        for j in range(img_GT.shape[1]):
            if np.array(img_GT)[i][j] == 255:
                mrk = 0
    if mrk == 1:
        return 0.0, 1000, 100
    lP = sitk.ReadImage(binary_GT) #pred
    lT = sitk.ReadImage(binary_R) #real
    labelTrue = lT
    labelPred = lP
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue, labelPred)
    aveHausdorff = hausdorffcomputer.GetAverageHausdorffDistance()
    hausdorff = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue, labelPred)
    h_dice = dicecomputer.GetDiceCoefficient()
    return h_dice, hausdorff, aveHausdorff

def metric_eval(pred_img, real_img):
    # step 1：读入图像，并灰度化
    img_GT = cv2.imread(pred_img, 0)
    img_R = cv2.imread(real_img, 0)

    ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret_R, binary_R = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    VOE = calVOE(binary_GT, binary_R)
    RVD = calRVD(binary_GT, binary_R)
    Precision = calPrecision(binary_GT, binary_R)
    Recall = calRecall(binary_GT, binary_R)
    Dice = calDice(binary_GT, binary_R)
    h_dice, hausdorff, aveHausdorff = calHausdorff(pred_img, real_img)
    metrics = [VOE, RVD, Precision, Recall, Dice, h_dice, hausdorff, aveHausdorff]
    return  metrics

def generate_respective_img(pred, true):
    color_table = [[238, 138, 91], [41, 95, 229], [100, 158, 74], [75, 90, 205], [95, 208, 248], [166, 112, 244],
                   [190, 142, 163]]
    pred_img = np.array(cv2.imread(pred))
    true_img = np.array(cv2.imread(true, 0))
    for c in range(7):
        pred_mask = np.zeros(true_img.shape)
        true_mask = np.zeros(true_img.shape)
        for h in range(true_img.shape[0]):
            for w in range(true_img.shape[1]):
                if pred_img[h, w, 0] == color_table[c][0] and pred_img[h, w, 1] == color_table[c][1] and pred_img[h, w, 2] == color_table[c][2]:
                    pred_mask[h, w] = 255
                if true_img[h, w] == (c+1)*30:
                    true_mask[h, w] = 255
        cv2.imwrite('pred' + str(c) + '.png', pred_mask)
        cv2.imwrite('true' + str(c) + '.png', true_mask)

    Metrics = np.zeros((7, 8))
    for c in range(7):
        metrics = metric_eval(
            'pred' + str(c) + '.png', 'true' + str(c) + '.png')
        print('dice(class-'+str(c+1)+'):', metrics[4])
        for j in range(Metrics.shape[1]):
            Metrics[c, j] += metrics[j]

    #for i in range(Metrics.shape[0]):
    #    Metrics[i] = Metrics[i]/7.0
    return Metrics

if __name__ == '__main__':
    metric_name = ['voe', 'rvd', 'precision', 'recall', 'dice', 'h_dice', 'haurdorff', 'ave_haurdorff']
    list_name=[0, 1, 2]
    Metrics = np.zeros((len(list_name), 7, 8))

    for i in range(len(list_name)):
        metrics = generate_respective_img('./ex_val_color_mask/'+str(list_name[i])+'.png', './ex_val_result/'+str(list_name[i])+'.png')
        for j in range(metrics.shape[0]):
            for k in range(metrics.shape[1]):
                Metrics[i, j, k] = metrics[j, k]

    for j in range(Metrics.shape[1]):
        for k in range(Metrics.shape[2]):
            print(metric_name[k]+'-mean(class-'+str(j+1)+'):', np.mean(Metrics[:, j, i]))
            print(metric_name[k]+'-std(class-'+str(j+1)+'):', np.std(Metrics[:, j, i]))
