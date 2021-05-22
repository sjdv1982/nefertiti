""" Randomly generate fragment trajectories
within an (optional) score threshold

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
from nefertiti.progressions.randombest import randombest as _randombest
from nefertiti.progressions.grow import init, init_coor_backbone
from nefertiti.progressions.coor import update_coordinates_backbone
from nefertiti.progressions.rmsd import (
    update_residuals_covar_backbone,
    rmsd_backbone,
)

def randombest_backbone_rmsd(
    refe, #str | np.ndarray,
    fraglib: np.ndarray,
    *,
    format: str,
    ntrajectories: int,
    max_rmsd: float,
    use_downstream_best: bool,
    minblocksize: int=10000,
    bb_atoms = ["N", "CA", "C", "O"],
    random_seed = 0,
) -> MainState:
    """
    Builds near-native fragment trajectories by random sampling
    using a fragment-duplicated backbone RMSD

    Input: 
    - a reference structure with nres residues.
      "format" indicates the format.
      - "pdb": in PDB format (text string)
      - "npy": as a shape=(nresidues, len(bb_atoms), 3) numpy array
    - a fragment library of shape (nfrag, fraglen, len(bb_atoms), 3)

    ntrajectories: the number of trajectories to generate.

    max_rmsd: maximum RMSD threshold for the generated trajectories

    use_downstream_best: if true, first obtain the downstream best RMSDs
     for fragment 2..N until the end. This will be very time-consuming for
    large structures. However, this will then prune the search very 
    efficiently. Recommended if both the structure and the max_rmsd are
    small.

    minblocksize: maximum storage space per stage. Affects memory requirements.
    Each stage has a size "maxsize", which is 2 x minblocksize, except for the
    last stage, where it is ntrajectories + minblocksize.
    
    What is stored at each stage N: 
    - an array of shape (maxsize, N, 3) fragment centers-of-mass
    - an array of shape (maxsize, N) trajectories (ints)
    - an array of shape (maxsize, 4, 4) matrices
    - an array of shape (maxsize, 3, 3) covariance matrices
    - an array of shape (maxsize,) residuals
    - an array of shape (maxsize, len(bb_atoms), 3) coordinates
    - an array of shape (maxsize, 3) scores

    bb_atoms: a list of backbone atoms. default: ["N", "CA", "C", "O"]

    Returns:
    The MainState object m. The interesting results are mostly in m.stages[-1]
    """
    np.random.seed(random_seed)

    from nefertiti.procedures.kbest import kbest_backbone_rmsd
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

    downstream_best = None    
    if use_downstream_best and fraglen > 1:
        refe2 = s.coor_residue.backbone[1:]
        ms0 = kbest_backbone_rmsd(
            refe2,
            fraglib, 
            format="npy",
            k=1
        )
        downstream_best = ms0.downstream_best_score

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
    maxsizes = [2*minblocksize] * nstages
    maxsizes[-1] = minblocksize + ntrajectories
    init(ms, nstages=nstages, maxsize=maxsizes, 
        with_coor=True,
        with_rmsd=True,
        with_all_matrices=False,
        swap_buffers=False
    )
    init_coor_backbone(ms, fraglen, len(bb_atoms))
    
    updaters = [
        update_coordinates_backbone,
        update_residuals_covar_backbone
    ]
    
    natoms = nstages * fraglen * len(bb_atoms)
    threshold = None
    if max_rmsd is not None:
        threshold = max_rmsd**2 * natoms
    _randombest(
        ms,
        scorer=rmsd_backbone,
        updaters=updaters,
        ntrajectories=ntrajectories,
        threshold=threshold,
        minblocksize=minblocksize,
        downstream_best=downstream_best
    )
    
    return ms