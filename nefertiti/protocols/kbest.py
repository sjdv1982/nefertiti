import numpy as np
from nefertiti.procedures.kbest import kbest_backbone_rmsd as _kbest_backbone_rmsd

def kbest_backbone_rmsd(
    refe, #str | np.ndarray,
    fraglib: np.ndarray,
    *,
    format: str,
    k: int
) -> tuple:
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
    Returns:
    - trajectories, a (poolsize, nresidues-fraglen+1) array of np.uint32
    - rmsds, a (poolsize) array of float
    """
    if isinstance(refe, str) and len(refe) < 200:
        raise ValueError("This function takes data content, not file names or URLs")

    main_state = _kbest_backbone_rmsd(
        refe, fraglib,
        format=format,
        k=k
    )
    natoms = main_state.refe.nfrags * main_state.refe.fraglen * len(main_state.refe.bb_atoms)
    last_stage = main_state.stages[-1]
    trajectories = last_stage.trajectories[:k]
    scores = last_stage.scores[:k]
    rmsds = np.sqrt(scores/natoms)
    return trajectories, rmsds