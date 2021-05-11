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
    traj = s.trajectories[size:newsize,-1]
    
    fraglib = s.fraglib
    fraglib_residuals = fraglib.coor.backbone_residuals
    traj_frag_residuals = fraglib_residuals[traj]
    
    refe = s.refe.coor_fragment.backbone4_centered
    refe_coor = refe[s.fragindex-1].reshape(-1, 4)[:, :3]
    
    covar = np.einsum("ikj,kl->ijl", coor, refe_coor)
    coor_shift = s.matrices[size:newsize, 3,:3]
    lc = len(s.bb_atoms) * s.fraglen
    residuals = traj_frag_residuals  + lc * (coor_shift * coor_shift).sum() 
    
    s.residuals[size:newsize] = residuals
    s.covar[size:newsize] = covar
    
def rmsd_backbone(
    s: Stage, newsize
) -> None:
    size = s.size

    refe = s.refe.coor_fragment
    refe_residual = refe.backbone_residuals[s.fragindex-1]

    covar = s.covar[size:newsize]
    coor_residuals = s.residuals[size:newsize] 
    residuals = coor_residuals + refe_residual

    natoms = s.fragindex * s.fraglen * len(s.bb_atoms)
    sd = superimpose_array_from_covar(covar, residuals, natoms, return_sd=True)
    s.scores[size:newsize] = sd

    '''
    # Local RMSD, direct
    coor = s.coor.backbone[size:newsize]
    coor = coor.reshape(len(coor), -1, 3)

    all_refe_coor = refe.coor_fragment.backbone4_centered
    refe_coor = all_refe_coor[s.fragindex-1].reshape(-1, 4)[:, :3]
    
    print((coor[12]*coor[12]).sum(), coor_residuals[12])
    #print(coor[12].T.dot(refe_coor), covar[12])
    #print(coor[12].mean(axis=0), refe_coor.mean(axis=0))
    #coor12 = coor[12] - coor[12].mean(axis=0)
    #print(coor12.T.dot(refe_coor), covar[12])

    rotmat_direct, rmsd_direct = superimpose_array(coor, refe_coor)
    print(rotmat_direct[0], rmsd_direct[:10])
    '''

    