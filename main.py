import os
from canny_edges import cannyEdge
import cv2 as cv

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