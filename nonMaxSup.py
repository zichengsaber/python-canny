import matplotlib.pyplot as plt
import numpy as np

'''
  File clarification:
    Find local maximum edge pixel using NMS along the line of the gradient
    - Input Mag: H x W matrix represents the magnitude of derivatives
    - Input Ori: H x W matrix represents the orientation of derivatives
    - Output M: H x W binary matrix represents the edge map after non-maximum suppression
'''
"""
(i-1,j-1),(i-1,j),(i-1,j+1)
(i,j-1),(i,j),(i,j+1)
(i+1,j-1),(i+1,j),(i+1,j+1)

|--->
|
v

"""
def choose(Mag,grad_Ori,i,j):
    if grad_Ori[i,j] in [0,180,-180]:
        grad1=Mag[i,j+1]
        grad2=Mag[i,j-1]
    elif grad_Ori[i,j] in [45,-135]:
        grad1=Mag[i-1,j+1]
        grad2=Mag[i+1,j-1]
    elif grad_Ori[i,j] in [90,-90]:
        grad1=Mag[i-1,j] 
        grad2=Mag[i+1,j]
    elif grad_Ori[i,j] in [135,-45]:
        grad1=Mag[i-1,j-1]
        grad2=Mag[i+1,j+1] #
    return grad1,grad2



def nonMaxSup(Mag,grad_Ori):
    H,W=Mag.shape
    suppressed = np.copy(Mag)
    suppressed.fill(0)
    # 遍历像素点
    for i in range(1,H-1): 
        for j in range(1,W-1):
            # 如果梯度为0
            if Mag[i,j]==0:
                suppressed[i,j]=0
            else:
                grad1,grad2=choose(Mag,grad_Ori,i,j)
                if grad1<=Mag[i,j] and grad2<=Mag[i,j]:
                    suppressed[i,j]=1
                else:
                    suppressed[i,j]=0
    """
    查看中间结果
    plt.figure()
    plt.imshow(np.uint8(suppressed))
    plt.set_cmap('gray')
    plt.colorbar()
    plt.savefig("./test/nms.png") 
    plt.figure()
    plt.imshow(np.uint8(suppressed*Mag))
    plt.set_cmap('gray')
    plt.colorbar()
    plt.savefig("./test/nms(1).png") 
    """
    
    return suppressed
