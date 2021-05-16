import numpy as np 

# high point edge Ori
def choose(x,y,edge_Ori):
    if edge_Ori[x,y] in [0,180,-180]:
        return (x,y-1,x,y+1)
    elif edge_Ori[x,y] in [45,-135]:
        return (x-1,y+1,x+1,y-1)
    elif edge_Ori[x,y] in [90,-90]:
        return (x-1,y,x+1,y)
    elif edge_Ori[x,y] in [135,-45]:
        return (x-1,y-1,x+1,y+1)

def edgeLink(M, Mag, edge_Ori):
    thresh_high=0.1*Mag.max()
    thresh_low=0.02*Mag.max()

    # 记录下梯度强度
    NMS=Mag*M 
    edge=np.copy(NMS)
    # 获取mask
    mask_high=edge>thresh_high
    mask_low=edge<thresh_low
    mask1,mask2=edge<=thresh_high,edge>=thresh_low
    # 一定包含
    edge[mask_high]=1
    # 不包含
    edge[mask_low]=0
    # maybe
    edge[mask1 & mask2]=-1

    # high index
    X,Y=np.where(mask_high)
    for i in range(len(X)):
        x1,y1,x2,y2=choose(X[i],Y[i],edge_Ori)
        if edge[x1,y1]==-1:
            mask_high[x1,y1]=True
        if edge[x2,y2]==-1:
            mask_high[x1,y1]=True
    
    edgeLinks=np.zeros(edge.shape)
    edgeLinks[mask_high]=255
    

    return edgeLinks
            


    
    

