import numpy as np
from nefertiti.procedures.greedy import greedy_backbone_rmsd as _greedy_backbone_rmsd

def greedy_backbone_rmsd(
    refe, #str | np.ndarray,
    fraglib: np.ndarray,
    *,
    format: str,
    poolsize: int
) -> tuple:
    """
    Builds near-native fragment trajectories in a greedy way
    using a fragment-duplicated backbone RMSD

    Input: 
    - a reference structure with nres residues.
      "format" indicates the format.
      - "pdb": in PDB format (text string)
      - "npy": as a shape=(nresidues, len(bb_atoms), 3) numpy array
    - a fragment library of shape (nfrag, fraglen, len(bb_atoms), 3)

    poolsize: size of the greedy pool 
    (keep the best <poolsize> RMSDs at each trajectory growing stage)

    Returns:
    - trajectories, a (poolsize, nresidues-fraglen+1) array of np.uint32
    - rmsds, a (poolsize) array of float
    """
    if isinstance(refe, str) and len(refe) < 200:
        raise ValueError("This function takes data content, not file names or URLs")
    main_state = _greedy_backbone_rmsd(
        refe, fraglib,
        format="pdb",
        poolsize=poolsize
    )
    natoms = main_state.refe.nfrags * main_state.refe.fraglen * len(main_state.refe.bb_atoms)
    last_stage = main_state.stages[-1]
    trajectories = last_stage.trajectories[:poolsize]
    scores = last_stage.scores[:poolsize]
    rmsds = np.sqrt(scores/natoms)
    return trajectories, rmsds