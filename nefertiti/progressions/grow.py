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
    *,
    with_coor: bool,
    with_rmsd: bool,
    with_all_matrices: bool,
    swap_buffers:bool,
) -> None:
    ms.stages = []
    if isinstance(maxsize, int):
        maxsize = [maxsize] * nstages
    else:
        assert len(maxsize) == nstages
    
    if swap_buffers:
        allmx = max(maxsize)
        trajs = [
            np.zeros((allmx, nstages),np.uint16),
            np.zeros((allmx, nstages),np.uint16),
        ]        
        if with_rmsd:
            all_fragcoms = [
                np.zeros((allmx, nstages, 3)),
                np.zeros((allmx, nstages, 3)),
            ]            
        if with_all_matrices:
            all_mats = [
                np.zeros((allmx, nstages, 4, 4)),
                np.zeros((allmx, nstages, 4, 4)),
            ]
        

    for n in range(nstages):
        stage = Stage()
        stage.parent = ms

        stage.size = 0
        mx = maxsize[n]
        stage.maxsize = mx
        stage.fragindex = n+1

        if with_coor:
            stage.nfrags = mx
            if swap_buffers and n > 1:
                stage.coor = ms.stages[-2].coor
            else:
                stage.coor = {}

        if mx is None:
            continue

        if swap_buffers:
            traj = trajs[n%2]
            stage.trajectories = traj[:mx, :n+1]
        else:
            stage.trajectories = np.zeros((mx, n+1),np.uint16)

        if swap_buffers and n > 1:
            stage.matrices = ms.stages[-2].matrices
            stage.scores = ms.stages[-2].scores
        else:
            stage.matrices = np.zeros((mx, 4, 4))
            stage.scores = np.zeros(mx)

        if with_rmsd:
            if swap_buffers:
                fragcoms = all_fragcoms[n%2]
                stage.fragcoms = fragcoms[:mx, :n+1]
            else:
                stage.fragcoms = np.zeros((mx, n+1, 3))

            if swap_buffers and n > 1:
                stage.covar = ms.stages[-2].covar
                stage.residuals = ms.stages[-2].residuals
            else:
                stage.covar = np.zeros((mx, 3, 3))
                stage.residuals = np.zeros(mx)
        
        if with_all_matrices:
            if swap_buffers:
                all_mat = all_mats[n%2]
                stage.all_matrices = all_mat[:mx, :n+1]
            else:
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
        ngrow = min(ngrow, stage1.size)
        traj = stage1.trajectories[:ngrow]
        traj_indices0 = cartesian_product((
            np.arange(ngrow),            
            np.arange(fraglibsize)
        ))
        traj_indices = np.empty((ngrow*fraglibsize, 3), int)
        ind = traj_indices0[:, 0]
        traj_indices[:, 0] = ind
        traj_indices[:, 1] = traj[:, -1][ind]
        traj_indices[:, 2] = traj_indices0[:, 1]
        mat = stage1.matrices[:ngrow]
        newmatrices = matmult(
            mat, transition_matrices, traj_indices
        )
        newtraj0 = traj[ind]
        newtraj = np.concatenate((newtraj0, traj_indices[:, 2,][:, None]), axis=1)
    
    ngrow2 = ngrow * fraglibsize
    newsize = stage2.size + ngrow2
    stage2.trajectories[stage2.size: newsize] = newtraj
    stage2.matrices[stage2.size:newsize] = newmatrices

    if stage1 is not None:
        if stage2.covar is not None:
            stage2.covar[stage2.size:newsize] = stage1.covar[ind]
        if stage2.residuals is not None:
            stage2.residuals[stage2.size:newsize] = stage1.residuals[ind]
        if stage2.fragcoms is not None:
            if stage1 is not None:
                stage2.fragcoms[stage2.size:newsize, :-1] = stage1.fragcoms[ind]

    if stage2.all_matrices is not None:
        if stage1 is not None:
            stage2.all_matrices[stage2.size:newsize, :-1] = stage1.all_matrices[ind]
        stage2.all_matrices[stage2.size:newsize, -1] = newmatrices

    for updater in updaters:
        updater(stage2, newsize)
    scorer(stage2, newsize)

    """
    # Works only for backbone RMSD growing...
    natoms = stage2.fragindex * len(stage2.bb_atoms) * stage2.fraglen 
    best, worst = stage2.scores[:newsize].min(), stage2.scores[:newsize].max()  
    print("!RMSD", np.sqrt(best/natoms), np.sqrt(worst/natoms))
    """

    stage2.size = newsize

    if stage1 is not None: 
        oldsize = stage1.size
        stage1.size -= ngrow
        size = stage1.size
        if size > 0:
            stage1.trajectories[:size] = stage1.trajectories[ngrow:oldsize]
            stage1.matrices[:size] = stage1.matrices[ngrow:oldsize]
            stage1.scores[:size] = stage1.scores[ngrow:oldsize]
            for attr in ("covar", "residuals", "fragcoms", "all_matrices"):
                v = getattr(stage1, attr)
                if isinstance(v, np.ndarray):
                    v[:size] = v[ngrow:oldsize]
            
            coor = stage1.coor
            if coor is not None:
                for attr in dir(coor):
                    v = getattr(coor, attr)
                    if isinstance(v, np.ndarray):
                        v[:size] = v[ngrow:oldsize]

def init_coor_backbone(ms, fraglen, bbsize):
    for stage in ms.stages:
        stage.coor.backbone = np.zeros((stage.maxsize,fraglen, bbsize, 3))

def sort_score(stage: Stage) -> None:
    """Sort the stage by score, lowest first"""
    assert stage.size
    size = stage.size
    scores = stage.scores[:size]
    ind = np.argsort(scores)
    stage.trajectories[:size] = stage.trajectories[ind]
    stage.matrices[:size] = stage.matrices[ind]
    stage.scores[:size] = stage.scores[ind]
    if stage.covar is not None:
        stage.covar[:size] = stage.covar[ind]
    if stage.residuals is not None:
        stage.residuals[:size] = stage.residuals[ind]
    if stage.fragcoms is not None:
        stage.fragcoms[:size] = stage.fragcoms[ind]
    if stage.all_matrices is not None:
        stage.all_matrices[:size] = stage.all_matrices[ind]
    if stage.coor is not None:
        for attr in dir(stage.coor):
            v = getattr(stage.coor, attr)
            if v is not None:
                v[:size] = v[ind]

def filter_score(stage: Stage) -> None:
    """Filters stage by score_threshold, if it has one
    Assumes that the stage has been sorted by score"""
    if stage.size and stage.score_threshold is not None:
        pos = np.searchsorted(
            stage.scores[:stage.size], 
            stage.score_threshold, 
            side='right'
        )
        if pos < stage.size:
            stage.size = pos

def filter_score_unsorted(stage: Stage, old_size) -> None:
    """Filters new trajectories in stage by score_threshold, if it has one
    Does not assume that the stage has been sorted by score
    Trajectories beyond old_size are assumed to be new"""
    if stage.size > old_size and stage.score_threshold is not None:
        mask = (stage.scores[old_size:stage.size] <= stage.score_threshold)
        masksum = mask.sum()
        if masksum == stage.size - old_size: # Nothing to do
            return
        if masksum == 0:
            stage.size = old_size
            return
        new_size = stage.size + masksum

        for attr in (
            "trajectories", "matrices", "scores", 
            "covar", "residuals", "fragcoms",
            "all_matrices"
        ):
            v = getattr(stage.coor, attr)
            if v is not None:
                v[old_size:new_size] = v[old_size:stage.size][mask]
        if stage.coor is not None:
            for attr in dir(stage.coor):
                v = getattr(stage.coor, attr)
                if v is not None:
                    v[old_size:new_size] = v[old_size:stage.size][mask]
        stage.size = new_size
