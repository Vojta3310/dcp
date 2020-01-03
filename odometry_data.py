import torch.utils.data as tdata
import pykitti
import numpy as np

import util


class OdometryDataset(tdata.Dataset):
    def __init__(self, basedir, sequence, n_points, kombine=100):
        self._data = pykitti.odometry(basedir, sequence)
        self._poses = self._data.poses
        self._n_points = n_points
        self._kombine = kombine
        vl = list(self._data.velo)
        self._velo=[]
        print("Randomizing data")
        for i in range(len(vl)-1):
          if np.size(vl[i],0)>0:
            self._velo.append(vl[i][np.random.choice(range(np.size(vl[i],0)),size=self._n_points).astype(int)])
        print("done randomizing data")
        
        
    def get(self, i, j):
      a_cloud = self._velo[i]#[: self._n_points]
      b_cloud = self._velo[j]#[: self._n_points]
      #a_cloud = self._velo[i][np.random.choice(range(np.size(self._velo[i],0)),size=self._n_points).astype(int)]
      #b_cloud = self._velo[j][np.random.choice(range(np.size(self._velo[j],0)),size=self._n_points).astype(int)]
      #print(i, "-", j)
    
      Ta=self._poses[i]
      Tb=self._poses[j]

      f=np.array([-1, 1, 1])
      axes=[2, 0, 1]
      indx=np.argsort(axes)
      a_cloud = a_cloud.T[indx, :] * f[:, np.newaxis]
      b_cloud = b_cloud.T[indx, :] * f[:, np.newaxis]
      
      T_ab=np.linalg.inv(Ta) @ Tb
      T_ba=np.linalg.inv(Tb) @ Ta
      
      return (
            a_cloud.astype("float32"),
            b_cloud.astype("float32"),
            T_ab.astype("float32"),
            T_ba.astype("float32"),
        )
      
    
    def __getitem__(self, i):
        j=i+1;
        if self._kombine>0:
          j=i%self._kombine
          i=round(i/self._kombine)
          if j>= i:
            j=j+1
        
        a_cloud, b_cloud, T_ab, T_ba=self.get(i,j)
        
        #print(a_cloud)
        
        a_R = T_ab[:3, :3]
        a_T = T_ab[:3, 3]
        b_R = T_ba[:3, :3]
        b_T = T_ba[:3, 3]
        
        E_ab = util.npmat2euler(a_R.reshape((1, 3, 3)), seq="xyz")
        E_ba = util.npmat2euler(b_R.reshape((1, 3, 3)), seq="xyz")
        
        return (
            a_cloud.astype("float32"),
            b_cloud.astype("float32"),
            a_R.astype("float32"),
            a_T.astype("float32"),
            b_R.astype("float32"),
            b_T.astype("float32"),
            E_ab.flatten().astype("float32"),
            E_ba.flatten().astype("float32"),
        )

    def __len__(self):
      if self._kombine>0:
        return len(self._velo)*self._kombine - 1
      
      return len(self._velo) - 1
