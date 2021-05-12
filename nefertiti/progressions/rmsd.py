"""Progressions related to RMSD calculation

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

import numpy as np

from ..MainState import Stage
from ..functions.superimpose import superimpose_array, superimpose_array_from_covar

def update_residuals_covar_backbone(
    s: Stage, newsize
) -> None:
    size = s.size
    coor = s.coor.backbone[size:newsize]
    coor = coor.reshape(len(coor), -1, 3)
    fragcom = s.matrices[size:newsize, 3,:3]
    coor_c = coor - fragcom[:, None]
        
    refe = s.refe.coor_fragment.backbone4_centered
    refe_coor = refe[s.fragindex-1].reshape(-1, 4)[:, :3]
    
    residuals = np.einsum("ijk,ijk->i", coor_c, coor_c)
    covar = np.einsum("ikj,kl->ijl", coor_c, refe_coor)    
    s.residuals[size:newsize] += residuals
    s.covar[size:newsize] += covar
    s.fragcoms[size:newsize, -1] = fragcom
    
def rmsd_backbone(
    s: Stage, newsize
) -> None:
    size = s.size

    refe = s.refe.coor_fragment
    refe_self_residual = refe.backbone_residuals[:s.fragindex].sum()
    nfragatoms = s.fraglen * len(s.bb_atoms)
    natoms = s.fragindex * nfragatoms

    refe_shifts = refe.backbone_com[:s.fragindex]
    refe_shifts = refe_shifts - refe_shifts.mean(axis=0)
    refe_shift_residual = nfragatoms * (refe_shifts * refe_shifts).sum()
    refe_residual = refe_self_residual + refe_shift_residual

    coor_self_residuals = s.residuals[size:newsize]
    coor_shifts = s.fragcoms[size:newsize]
    coor_shifts = coor_shifts - coor_shifts.mean(axis=1)[:, None]
    coor_shift_residuals = nfragatoms * np.einsum("ijk,ijk->i",coor_shifts,coor_shifts)
    coor_residuals = coor_self_residuals + coor_shift_residuals

    covar_raw = s.covar[size:newsize]
    covar_shift0 = coor_shifts[:, :, :, None] * refe_shifts[None, :, None, :]
    covar_shift = covar_shift0.sum(axis=1) * nfragatoms 
    covar = covar_raw + covar_shift

    residuals = coor_residuals + refe_residual
    
    sd = superimpose_array_from_covar(covar, residuals, natoms, return_sd=True)
    s.scores[size:newsize] = sd

    """
    # Check (up to first 2 fragments)
    if s.fragindex <= 2:

        rotmat, rmsd = superimpose_array_from_covar(covar, residuals, natoms, return_sd=False)
        print(rotmat[0], rmsd[:10])

        all_refe_coor = refe.coor_fragment.backbone4_centered
        this_refe_coor = all_refe_coor[:s.fragindex].reshape(-1, 4)[:, :3]
        this_refe_coor = this_refe_coor - this_refe_coor.mean(axis=0)

        coor = s.coor.backbone[size:newsize]
        coor = coor.reshape(len(coor), -1, 3)
        if s.fragindex == 2:
            fraglib = s.parent.fraglib
            t = s.trajectories[size:newsize, 0]         
            old_coor = fraglib.coor.backbone4_centered[t][:, :, :, :3]   
            old_coor = old_coor.reshape(len(old_coor), -1, 3)
            coor = np.concatenate((old_coor, coor), axis=1)
        coor = coor - coor.mean(axis=1)[:, None] 
        rotmat_direct, rmsd_direct = superimpose_array(coor, this_refe_coor)
        print(rotmat_direct[0], rmsd_direct[:10])
    """