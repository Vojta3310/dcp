from mpl_toolkits import mplot3d


import numpy as np
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






data=odometry_data.OdometryDataset("dataset/", "00", 10000)
i=0
draw_pointCloud(data[i][0], data[i][1], data[i][2], data[i][3])

exit()

fig = plt.figure()

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');


plt.show()
