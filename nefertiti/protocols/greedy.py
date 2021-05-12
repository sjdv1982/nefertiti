""" Builds fragment trajectories in a greedy way

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

import numpy as np

from nefertiti.MainState import (
    MainState, Stage, FragmentLibrary, StructureRepresentation
)
from nefertiti.progressions.prepare_backbone import (
    prepare_backbone, prepare_backbone_from_pdb
)
from nefertiti.progressions.fraglib import load_backbone_fraglib
from nefertiti.progressions.greedy import greedy as _greedy
from nefertiti.progressions.grow import init, init_coor_backbone
from nefertiti.progressions.coor import update_coordinates_backbone
from nefertiti.progressions.rmsd import (
    update_residuals_covar_backbone,
    rmsd_backbone,
)

def greedy_backbone_rmsd(
    refe, #str | np.ndarray,
    fraglib: np.ndarray,
    *,
    format: str,
    poolsize: int,
    chunksize: int=100000,
    bb_atoms = ["N", "CA", "C", "O"]
) -> MainState:
    """
    Builds near-native fragment trajectories in a greedy way
    using a fragment-duplicated backbone RMSD

    Input: 
    - a reference structure with nres residues.
      "format" indicates the format.
      - "pdb": in PDB format (text string)
      - "npy": as a shape=(natoms, 3) numpy array
    - a fragment library of shape (nfrag, fraglen, len(bb_atoms), 3)

    poolsize: size of the greedy pool 
    (keep the best <poolsize> RMSDs at each trajectory growing stage)

    chunksize: maximum storage space per stage. Affects memory requirements:
    What is stored at each stage N: 
    - an array of shape (chunksize, N, 3) fragment centers-of-mass
    - an array of shape (chunksize, N) trajectories (ints)
    - an array of shape (chunksize, 4, 4) matrices
    - an array of shape (chunksize, 3, 3) covariance matrices
    - an array of shape (chunksize,) residuals
    - an array of shape (chunksize, len(bb_atoms), 3) coordinates
    - an array of shape (chunksize, 3) scores
    Normally, N goes from 1 to nres-fraglen+1, but buffers are swapped,
     so there are only 2 independent stages.

    bb_atoms: a list of backbone atoms. default: ["N", "CA", "C", "O"]

    Returns:
    The MainState object m. The interesting results are mostly in m.stages[-1]
    """
    if chunksize < poolsize:
        raise ValueError("chunksize must be at least poolsize")

    fraglen = fraglib.shape[1]

    assert fraglib.shape[2] == len(bb_atoms), (fraglib.shape, bb_atoms)
    
    s = StructureRepresentation()
    s.bb_atoms = bb_atoms
    if format == "pdb":
        prepare_backbone_from_pdb(s, refe)
    elif format == "npy":
        prepare_backbone(s, refe)
    else:
        raise ValueError(format)
    
    nres = s.nresidues
    if fraglen > nres:
        raise ValueError("Cannot grow reference structures shorter than the size of a fragment")
    nstages = nres - fraglen + 1

    ms = MainState()
    ms.fraglen = fraglen
    ms.refe = s
    ms.nfrags = nstages
    ms.bb_atoms = bb_atoms
    f = FragmentLibrary()
    f.nfrags = len(fraglib)  # Here: fragment library size
    ms.fraglib = f
    load_backbone_fraglib(f, fraglib)
    init(ms, nstages=nstages, maxsize=chunksize, 
        with_coor=True,
        with_rmsd=True,
        with_all_matrices=False,
        swap_buffers=True
    )
    init_coor_backbone(ms, fraglen, len(bb_atoms))
    
    updaters = [
        update_coordinates_backbone,
        update_residuals_covar_backbone
    ]
    _greedy(
        ms,
        scorer=rmsd_backbone,
        updaters=updaters,
        poolsize=poolsize
    )
    
    return ms