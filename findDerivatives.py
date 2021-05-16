import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
from numpy.core.arrayprint import _guarded_repr_or_str

'''
  File clarification:
    Compute gradient put ginformation of the inrayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the derivatives along x-axis
    - Output Magy: H x W matrix represents the derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''
def findDerivatives(I_gray):
    
    # smoothing kernels
    gaussian = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]],dtype=np.float32) 
    gaussian=gaussian/gaussian.sum()

    # kernel for x and y gradient
    # Sobel 卷积算子
    dx = np.float32([
      [-1.0, 0.0, 1.0], 
      [-2.0, 0.0, 2.0], 
      [-1.0, 0.0, 1.0]
    ])
    dy = np.float32([
      [1.0, 2.0, 1.0], 
      [0.0, 0.0, 0.0], 
      [-1.0, -2.0, -1.0]
    ])
    
    ###############################################################################
    # Your code here: calculate the gradient magnitude and orientation
    ###############################################################################
    
    # 利用高斯梯度对图像卷积
    dx_G=cv.filter2D(gaussian,-1,dx)
    dy_G=cv.filter2D(gaussian,-1,dy)
    # 计算Ix,Iy 注意filter计算要求float32
    Magx=cv.filter2D(np.float32(I_gray),-1,dx_G)
    Magy=cv.filter2D(np.float32(I_gray),-1,dy_G)
    
    # 计算梯度强度
    Mag=np.sqrt(Magx*Magx+Magy*Magy)
    
    np.clip(Mag,0,255)
    # 计算边缘梯度的方向
    eps=1e-8
    Ori_g=np.arctan(Magy/(Magx+eps))
    # 计算边缘方向
    Ori_e=np.arctan(-Magx/(Magy+eps))
    # 查看结果
    
    

    """
    plt.figure()
    plt.imshow(np.uint8(np.round(Mag)))
    plt.set_cmap('gray')
    plt.colorbar()
    plt.savefig("./test/Im(2).png")
    plt.figure()
    plt.imshow(np.uint8(np.round(Mag)))
    plt.set_cmap('hot')
    plt.colorbar()
    plt.savefig("./test/Im(1).png")
    plt.figure()
    plt.imshow(np.uint8(np.round(Magx)))
    plt.set_cmap('gray')
    plt.colorbar()
    plt.savefig("./test/Ix(1).png")
    plt.figure()
    plt.imshow(np.uint8(np.round(Magy)))
    plt.set_cmap('gray')
    plt.colorbar()
    plt.savefig("./test/Iy(1).png")
    """

    return Mag,Magx,Magy,Ori_g,Ori_e
    
