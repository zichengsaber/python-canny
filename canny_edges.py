import numpy as np
import cv2 as cv 
from findDerivatives import findDerivatives
from helper import get_discrete_orientation
from nonMaxSup import nonMaxSup
from edgeLink import edgeLink
import matplotlib.pyplot as plt
import os
def cannyEdge(I):
    # convert RGB image to gray color space
    im_gray=cv.cvtColor(I,cv.COLOR_BGR2GRAY)
    # 计算梯度
    Mag,Magx, Magy, Ori_g,Ori_e = findDerivatives(im_gray)
    # 获得离散化的方向
    grad_Ori, edge_Ori = get_discrete_orientation(Ori_g,Ori_e)
    # 非极大值抑制
    M = nonMaxSup(Mag, grad_Ori)
    # 边缘连接
    E = edgeLink(M, Mag, edge_Ori)

    return E
    
 


if __name__ =="__main__":
    img_names=os.listdir('./img')

    print(img_names)
    for name in img_names:
        path=os.path.join('img',name)
        img=cv.imread(path)
        edge=cannyEdge(img)
        img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # 对比
        cv.imwrite("./result/canny-{}".format(name),edge)
        
    
    
    
    
    
    


