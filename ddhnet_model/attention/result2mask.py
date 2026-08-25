import cv2
import os
import numpy as np

result_pth = './ex_val_result/'
out_pth ='./ex_val_mask/'
img_list = os.listdir(result_pth)

if not os.path.exists(out_pth):
    os.makedirs(out_pth)
for i in range(1, 8):
    if not os.path.exists(out_pth+str(i)):
        os.makedirs(out_pth+str(i))

for i in range(len(img_list)):
    if img_list[i][-3:]  == 'png':
        img = np.array(cv2.imread(result_pth+img_list[i], 0))
        mask = [np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape)]
        for w in range(img.shape[0]):
            for h in range(img.shape[1]):
                for c in range(1, 8):
                    if img[w, h] == c*30:
                        mask[c-1][w, h] = 255

        for c in range(len(mask)):
            print(out_pth+str(c+1)+'/'+img_list[i][:-4]+'_'+str(c)+'.png')
            cv2.imwrite(out_pth+str(c+1)+'/'+img_list[i][:-4]+'_'+str(c)+'.png', mask[c])
