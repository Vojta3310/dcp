from mpl_toolkits import mplot3d


import numpy as np
import torch
import matplotlib.pyplot as plt


import odometry_data
import torch.utils.data as tdata
import pykitti

import util



def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    



def draw_pointCloud(ax, cloud, T, c):

  a_cloud= np.dot(T,np.r_[cloud,[np.ones(np.shape(cloud)[1])]])
  
  ax.scatter3D(list(a_cloud[0]), list(a_cloud[1]), list(a_cloud[2]), s=.001,c=c, marker='o', zorder=3);
  

def draw_Sequence(data, net):
  fig = plt.figure()
  ax = plt.gca(projection='3d')
  ax = plt.axes(projection='3d', proj_type = 'ortho')
  ax.view_init(0, -90)


  T=np.identity(4)
  Tp=np.identity(4)
  for i in range(len(data)):
    a_cloud=data[i][0]
    b_cloud=data[i][1]
    R_ab = data[i][2]
    T_ab = data[i][3]
    
    Ti=np.c_[R_ab,[T_ab[0],T_ab[1],T_ab[2]]]
    Ti=np.r_[Ti,[[0,0,0,1]]]
    
    T=T @ Ti
    
    O=T @ [0, 0, 1, 1]
    
    if i%20==0:
      draw_pointCloud(ax,b_cloud.astype("float32"), T.astype("float32"),'b')
    ax.scatter3D(O[0], O[1], O[2],c='r', marker='^',s=0.5, zorder=1)
  
    if net is None:
      continue 
    
    (R_ab_p, T_ab_p, R_ba_p, T_ba_p) = net(torch.from_numpy(np.array([a_cloud])), torch.from_numpy(np.array([b_cloud])))
    R_ab_p=R_ab_p.detach().numpy()
    T_ab_p=T_ab_p.detach().numpy()
    print(R_ab_p)
    print(T_ab_p)
    #torch.save(R_ab_p, "checkpoints/data/R_ab_p_" + str(i) + ".pt")
    #torch.save(T_ab_p, "checkpoints/data/T_ab_p_" + str(i) + ".pt")
    
    #R_ab_p=torch.load("checkpoints/data/R_ab_p_" + str(i) + ".pt")
    #T_ab_p=torch.load("checkpoints/data/T_ab_p_" + str(i) + ".pt")
    
    
    
    
    Ti=np.c_[R_ab_p[0],[T_ab_p[0][0],T_ab_p[0][1],T_ab_p[0][2]]]
    Ti=np.r_[Ti,[[0,0,0,1]]]
    
    Tp=Tp @ Ti
    
    O=Tp @ [0, 0, 0, 1]
    
    if i%20==0:
      draw_pointCloud(ax,b_cloud.astype("float32"), Tp.astype("float32"),'c')
    ax.scatter3D(O[0], O[1], O[2],c='m', marker='^',s=0.5)
    
  
  set_axes_equal(ax)
  plt.show()



if __name__ == "__main__":
  data=odometry_data.OdometryDataset("dataset/", "00", 1000, kombine=0)
  draw_Sequence(data, None)

