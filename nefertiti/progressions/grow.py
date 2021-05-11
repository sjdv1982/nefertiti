"""Progressions to grow fragment trajectories by one fragment,
adding them t

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""
import numpy as np
from typing import List
from ..MainState import MainState, Stage
from ..functions.matrix import matmult

# From https://stackoverflow.com/a/11146645
def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T
    
def init(
    ms: MainState, 
    nstages: int,
    maxsize, #int | List[int],
    with_coor: bool,
    with_rmsd: bool,
    with_all_matrices: bool,
) -> None:
    ms.stages = []
    if isinstance(maxsize, int):
        maxsize = [maxsize] * nstages
    else:
        assert len(maxsize) == nstages
    for n in range(nstages):
        stage = Stage()
        stage.parent = ms

        stage.size = 0
        mx = maxsize[n]
        stage.maxsize = mx
        stage.fragindex = n+1

        if with_coor:
            stage.nfrags = mx
            stage.coor = {}

        if mx is None:
            continue

        stage.trajectories = np.zeros((mx, n+1),np.uint16)
        stage.matrices = np.zeros((mx, 4, 4))
        stage.scores = np.zeros(mx)

        if with_rmsd:
            stage.covar = np.zeros((mx, 3, 3))
            stage.residuals = np.zeros(mx)
        
        if with_all_matrices:
            stage.all_matrices = np.zeros((mx, n+1, 4, 4))

        ms.stages.append(stage)    

def grow(stage1: Stage, stage2: Stage, *, scorer, updaters):    
    space = stage2.maxsize - stage2.size
    fraglib = stage2.fraglib
    transition_matrices = fraglib.matrices
    fraglibsize = fraglib.nfrags
    assert space >= fraglibsize
    ngrow = space // fraglibsize    
    if stage1 is None:
        ngrow = 1
        newtraj = np.arange(fraglibsize)[:, None]
        newmatrices = np.eye(4)
    else:  
        ngrow = min(ngrow, len(stage1.trajectories))
        traj = stage1.trajectories[:ngrow]
        traj_indices = cartesian_product(
            np.arange(ngrow),
            np.arange(fraglibsize)
        )
        mat = stage1.matrices[:ngrow]
        newmatrices = matmult(
            mat, transition_matrices, traj_indices
        )
        newtraj0 = traj[traj_indices[:, 0]]
        newtraj = np.concatenate(newtraj0, traj_indices[:, 1], axis=1)
    
    ngrow2 = ngrow * fraglibsize
    newsize = stage2.size + ngrow2
    stage2.trajectories[stage2.size: newsize] = newtraj
    stage2.matrices[stage2.size:newsize] = newmatrices

    if stage2.all_matrices is not None:
        if stage1 is not None:
            stage2.all_matrices[stage2.size:newsize, :-1] = stage1.all_matrices
        stage2.all_matrices[stage2.size:newsize, -1] = newmatrices

    for updater in updaters:
        updater(stage2, newsize)
    scorer(stage2, newsize)

    stage2.size = newsize

    if stage1 is not None:
        stage1.size -= ngrow
        stage1.trajectories[:-ngrow] = stage1.trajectories[ngrow:]
        stage1.matrices[:-ngrow] = stage1.matrices[ngrow:]
        stage1.scores[:-ngrow] = stage1.scores[ngrow:]
        coor = stage1.coor
        if coor is not None:
            for attr in dir(coor):
                v = getattr(coor, attr)
                if isinstance(v, np.ndarray):
                    v[:-ngrow] = v[ngrow:]

def init_coor_backbone(ms, fraglen, bbsize):
    for stage in ms.stages:
        stage.coor.backbone = np.zeros((stage.maxsize,fraglen, bbsize, 3))
