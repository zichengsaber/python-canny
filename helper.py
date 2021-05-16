import numpy as np
import matplotlib.pyplot as plt
def get_gradient_angle(angleDeg):
    # 找一个最靠近的方向
    discrete = [0, 45, 90, 135, 180, -45, -90, -135, -180]
    dir = min(discrete, key=lambda x:abs(x-angleDeg))

    return dir


def get_edge_angle(a):
    discrete = [0, 45, 90, 135, 180, -45, -90, -135, -180]
    dir = min(discrete, key=lambda x:abs(x-a))

    return dir


def get_discrete_orientation(Ori_g,Ori_e):
    angle_Ori_g= np.degrees(Ori_g)
    angle_Ori_e= np.degrees(Ori_e)
    # 函数向量化
    get_gradient_angle_vect = np.vectorize(get_gradient_angle)
    # 离散化的梯度方向
    discrete_gradient_orientation = get_gradient_angle_vect(angle_Ori_g)

    get_edge_angle_vect = np.vectorize(get_edge_angle)
    # 离散化的边缘方向
    # 需要删除np.absolute
    discrete_edge_orientation = get_edge_angle_vect(angle_Ori_e)
    

    return discrete_gradient_orientation, discrete_edge_orientation

if __name__=="__main__":
    # 测试 30,60,-60,-45,-15
    test=np.array([np.pi/6,np.pi/3,-np.pi/3,-np.pi/4,-np.pi/12])
    test=np.degrees(test)
    get_gradient_angle_vect = np.vectorize(get_gradient_angle)
    test=get_gradient_angle_vect(test)
    print(test)

