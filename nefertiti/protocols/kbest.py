"""Enumerates the k-best fragment trajectories

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

from nefertiti.protocols.greedy import greedy_backbone_rmsd
import numpy as np

from nefertiti.MainState import (
    MainState, Stage, FragmentLibrary, StructureRepresentation
)
from nefertiti.progressions.prepare_backbone import (
    prepare_backbone, prepare_backbone_from_pdb,
    select_last_backbone
)
from nefertiti.progressions.fraglib import load_backbone_fraglib
from nefertiti.progressions.kbest import kbest as _kbest
from nefertiti.progressions.greedy import greedy as _greedy
from nefertiti.progressions.grow import init, init_coor_backbone
from nefertiti.progressions.coor import update_coordinates_backbone
from nefertiti.progressions.rmsd import (
    update_residuals_covar_backbone,
    rmsd_backbone,
)

def kbest_backbone_rmsd(
    refe, #str | np.ndarray,
    fraglib: np.ndarray,
    *,
    format: str,
    k: int,
    maxblocksize: int=50000,
    minblocksize: int=200,
    bb_atoms = ["N", "CA", "C", "O"]
) -> MainState:
    """
    Enumerates the k-best near-native fragment trajectories
    using a fragment-duplicated backbone RMSD
    Input: 
    - a reference structure with nres residues.
      "format" indicates the format.
      - "pdb": in PDB format (text string)
      - "npy": as a shape=(nresidues, len(bb_atoms), 3) numpy array
    - a fragment library of shape (nfrag, fraglen, len(bb_atoms), 3)

    k: the best trajectories to keep

    minblocksize: minimum block size to send to the next stage.
    Keep this small to reach an initial estimate soon (leading to efficient pruning)
    Don't make it too small (minblocksize * nfrag should be in the thousands)
    to keep Python function call overhead reasonable.

    maxblocksize: maximum storage space per stage. Affects memory requirements:
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
    assert minblocksize < maxblocksize

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
    s_copy = s.copy()

    nres = s.nresidues
    if fraglen > nres:
        raise ValueError("Cannot grow reference structures shorter than the size of a fragment")
    nstages = nres - fraglen + 1

    ms0 = MainState()
    ms0.fraglen = fraglen
    f = FragmentLibrary()
    f.nfrags = len(fraglib)  # Here: fragment library size
    ms0.fraglib = f
    load_backbone_fraglib(f, fraglib)
    ms0.bb_atoms = bb_atoms
    ms_copy = ms0.copy()

    updaters = [
        update_coordinates_backbone,
        update_residuals_covar_backbone
    ]
    def copy_s():
        s = s_copy.copy()
        s.coor_residue = s_copy.coor_residue.copy()
        s.coor_fragment = s_copy.coor_fragment.copy()
        return s

    ms = None
    def _greedy_iter(stagelen):
        nonlocal ms
        ms = ms_copy.copy()
        s = copy_s()

        ms.refe = s
        ms.nfrags = stagelen
        select_last_backbone(s, stagelen)

        init(ms, nstages=stagelen, 
            maxsize=maxblocksize, 
            with_coor=True,
            with_rmsd=True,
            with_all_matrices=False,
            swap_buffers=True
        )
        init_coor_backbone(ms, fraglen, len(bb_atoms))
        
        _greedy(
            ms,
            scorer=rmsd_backbone,
            updaters=updaters,
            poolsize=200
        )

    def _best_iter(stagelen, greedy_best, best_msds):
        assert len(best_msds) == stagelen - 1
        nonlocal ms
        ms = ms_copy.copy()
        s = copy_s()

        ms.refe = s
        select_last_backbone(s, stagelen)
        ms.nfrags = stagelen

        init(ms, nstages=stagelen, 
            maxsize=maxblocksize, 
            with_coor=True,
            with_rmsd=True,
            with_all_matrices=False,
            swap_buffers=False
        )
        init_coor_backbone(ms, fraglen, len(bb_atoms))
        
        upper_estimate=None
        if greedy_best is not None:
            upper_estimate=greedy_best+0.001 #in case of rounding errors
        _kbest(
            ms,
            scorer=rmsd_backbone,
            updaters=updaters,
            k=1,
            minblocksize=minblocksize,
            upper_estimate=upper_estimate,
            downstream_best=best_msds,
        )

    best_msds = []
    tot_nfrags = s_copy.nfrags
    for n in range(tot_nfrags-1):
        print("PRECALC {}/{}".format(n+1, tot_nfrags-1))
        _greedy_iter(n+1)
        natoms = ms.nfrags * ms.fraglen * len(ms.bb_atoms)
        greedy_best = ms.stages[-1].scores[0]        
        print("GBEST", n+1, greedy_best, np.sqrt(greedy_best/natoms))
        _best_iter(n+1, greedy_best, best_msds)
        best = ms.stages[-1].scores[0]
        print("BEST ", n+1, best, np.sqrt(best/natoms))
        best_msds.insert(0, best)
        print()

    upper_estimate = None
    if k == 1:
        _greedy_iter(tot_nfrags)
        natoms = ms.nfrags * ms.fraglen * len(ms.bb_atoms)
        greedy_best = ms.stages[-1].scores[0]
        print("GBEST", tot_nfrags, greedy_best, np.sqrt(greedy_best/natoms))
        upper_estimate=greedy_best+0.001 #in case of rounding errors

    ms.refe = s_copy.copy()
    ms.nfrags = tot_nfrags

    maxblocksizes = [maxblocksize] * tot_nfrags
    maxblocksizes[-1] += k
    init(ms, nstages=tot_nfrags, 
        maxsize=maxblocksizes, 
        with_coor=True,
        with_rmsd=True,
        with_all_matrices=False,
        swap_buffers=False
    )
    init_coor_backbone(ms, fraglen, len(bb_atoms))

    _kbest(
        ms,
        scorer=rmsd_backbone,
        updaters=updaters,
        k=k,
        minblocksize=minblocksize,
        upper_estimate=upper_estimate,
        downstream_best=best_msds,
    )

    return ms