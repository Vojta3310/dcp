from mpl_toolkits import mplot3d


import numpy as np
import torch
import matplotlib.pyplot as plt


import odometry_data

def draw_pointCloud(a_cloud, b_cloud, R_ab, T_ab):
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  
  R=np.identity(3)#)R_ab[:,:]
  R[0]=R_ab[2,:]
  R[1]=R_ab[0,:]
  R[2]=R_ab[1,:]
  R=R+np.identity(3)
  b_cloud= np.dot(R,b_cloud)
  
  ax.scatter3D(list(a_cloud[0]), list(a_cloud[1]), list(a_cloud[2]),c='g', marker='o');
  ax.scatter3D(list(b_cloud[0]+T_ab[2]), list(b_cloud[1]+T_ab[0]), list(b_cloud[2]+T_ab[1]),c='b', marker='o');
  
  plt.show()



def draw_Sequence(data, net):
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  
  
  a_cloud=data[0][0]
  ax.scatter3D(list(a_cloud[0]), list(a_cloud[1]), list(a_cloud[2]),c='g', marker='o');
  
  T=np.identity(4)
  Tp=np.identity(4)
  for i in range(4):#range(len(data)):
    a_cloudO=data[i][0]
    b_cloudO=data[i][1]
    R_ab = data[i][2]
    T_ab = data[i][3]
    
    R=np.identity(3)
    R[0]=R_ab[2,:]
    R[1]=R_ab[0,:]
    R[2]=R_ab[1,:]
    R=R+np.identity(3)
    Ti=np.c_[R,[T_ab[2],T_ab[0],T_ab[1]]]
    Ti=np.r_[Ti,[[0,0,0,1]]]
    
    T=T @ Ti
    
    O=T @ [0, 0, 0, 1]    
    b_cloud= np.dot(T,np.r_[b_cloudO,[np.ones(np.shape(b_cloudO)[1])]])
    
    ax.scatter3D(list(b_cloud[0]), list(b_cloud[1]), list(b_cloud[2]),c='b', marker='o');
    ax.scatter3D(O[0], O[1], O[2],c='r', marker='^')
    
    #continue 
    
    (R_ab_p, T_ab_p, R_ba_p, T_ba_p,) = net(torch.from_numpy(np.array([a_cloudO])), torch.from_numpy(np.array([b_cloudO])))
    
    
    
    R=np.identity(3)
    R[0]=R_ab_p[0,2,:].detach().numpy()
    R[1]=R_ab_p[0,0,:].detach().numpy()
    R[2]=R_ab_p[0,1,:].detach().numpy()
    #R=R+np.identity(3)
    Ti=np.c_[R,[T_ab_p[0,2].detach().numpy(),T_ab_p[0,0].detach().numpy(),T_ab_p[0,1].detach().numpy()]]
    Ti=np.r_[Ti,[[0,0,0,1]]]
    
    Tp=Tp @ Ti
    print(Tp)
    
    O=Tp @ [0, 0, 0, 1]    
    b_cloud= np.dot(Tp,np.r_[b_cloudO,[np.ones(np.shape(b_cloudO)[1])]])
    
    ax.scatter3D(list(b_cloud[0]), list(b_cloud[1]), list(b_cloud[2]),c='c', marker='o');
    ax.scatter3D(O[0], O[1], O[2],c='m', marker='^')
    
  plt.show()



#data=odometry_data.OdometryDataset("dataset/", "00", 10000)
#draw_Sequence(data, 0)
#i=0
#draw_pointCloud(data[i][0], data[i][1], data[i][2], data[i][3])

