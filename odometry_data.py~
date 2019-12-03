import torch.utils.data as tdata
import pykitti

import util


class OdometryDataset(tdata.Dataset):
    def __init__(self, basedir, sequence, n_points):
        self._data = pykitti.odometry(basedir, sequence)
        self._velo = list(self._data.velo)
        self._poses = self._data.poses
        self._n_points = n_points

    def __getitem__(self, i):
        a_cloud = self._velo[i][: self._n_points]
        b_cloud = self._velo[i + 1][: self._n_points]
        a_R = self._poses[i][:3, :3]
        a_T = self._poses[i][:3, 3]
        b_R = self._poses[i + 1][:3, :3]
        b_T = self._poses[i + 1][:3, 3]

        R_ab = b_R - a_R
        R_ba = -R_ab
        E_ab = util.npmat2euler(R_ab.reshape((1, 3, 3)), seq="xyz")
        E_ba = util.npmat2euler(R_ba.reshape((1, 3, 3)), seq="xyz")
        T_ab = b_T - a_T
        T_ba = -T_ab

        a_cloud = a_cloud.T[:3, :]
        b_cloud = b_cloud.T[:3, :]

        return (
            a_cloud.astype("float32"),
            b_cloud.astype("float32"),
            R_ab.astype("float32"),
            T_ab.astype("float32"),
            R_ba.astype("float32"),
            T_ba.astype("float32"),
            E_ab.flatten().astype("float32"),
            E_ba.flatten().astype("float32"),
        )

    def __len__(self):
        return len(self._data) - 1
