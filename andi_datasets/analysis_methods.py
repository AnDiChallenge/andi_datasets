# AUTOGENERATED! DO NOT EDIT! File to edit: source_nbs/analysis_methods.ipynb (unless otherwise specified).

__all__ = ['get_changepoints_convexhull']

# Cell
import numpy as np

# Cell
from scipy.spatial import ConvexHull

def get_changepoints_convexhull(trajs, tau = 10):
    CPs = []
    for traj in trajs:
        traj = np.array(traj)

        Sd = np.zeros(traj.shape[0]-2*tau)
        for k in range(traj.shape[0]-2*tau):
            Sd[k] = ConvexHull(traj[k:(k+2*tau)]).volume

        below_mean = Sd < Sd.mean()
        cp_traj = np.argwhere(below_mean[1:] != below_mean[:-1])+1
        CPs.append(cp_traj+tau)

    return CPs