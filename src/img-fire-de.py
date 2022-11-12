import cv2 
import numpy as np
import pandas as pd
import os
os.chdir("D:/Document/专业课类/数字图像处理/Project/assets/")

img = cv2.imread('fire2.jpg')
cv2.namedWindow("RESULT",cv2.WINDOW_NORMAL)
redThre = 135  # 115~135红色分量阈值
sThre = 60  # 55~65饱和度阈值

B = img[:, :, 0]
G = img[:, :, 1]
R = img[:, :, 2]

B1 = img[:, :, 0] / 255
G1 = img[:, :, 1] / 255
R1 = img[:, :, 2] / 255
# minValue = np.array(
#     np.where(R1 <= G1, np.where(G1 <= B1, R1, np.where(R1 <= B1, R1, B1)), np.where(G1 <= B1, G1, B1)))
minValue = np.array(
    np.where(R1 <= G1, np.where(R1 <= B1, R1, B1),np.where(G1 <= B1, G1, B1))
)
sumValue = R1 + G1 + B1
# HSI中S分量计算公式
S = np.array(np.where(sumValue != 0, (1 - 3.0 * minValue / sumValue), 0))
Sdet = (255 - R) / 20
SThre = ((255 - R) * sThre / redThre)
#判断条件
fireImg = np.array(
    np.where(R > redThre, np.where(R >= G, np.where(G >= B, np.where(S > 0, np.where(S > Sdet, np.where(
        S >= SThre, 255, 0), 0), 0), 0), 0), 0))

gray_fireImg = np.zeros([fireImg.shape[0], fireImg.shape[1], 1], np.uint8)
gray_fireImg[:, :, 0] = fireImg
meBImg = cv2.medianBlur(gray_fireImg, 5)
kernel = np.ones((5, 5), np.uint8)
ProcImg = cv2.dilate(meBImg, kernel)
#绘制矩形框
contours, _ = cv2.findContours(ProcImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ResImg = img.copy()
for c in range(0, len(contours)):
    # 获取矩形的左上角坐标(x,y)，以及矩形的宽和高w、h
    x, y, w, h = cv2.boundingRect(contours[c])
    l_top = (x, y)
    r_bottom = (x + w, y + h)
    cv2.rectangle(ResImg, l_top, r_bottom, (255, 0, 0), 2)
cv2.imshow("RESULT", ResImg)
c = cv2.waitKey(0)
