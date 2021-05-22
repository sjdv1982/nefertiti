import numpy as np
from nefertiti.procedures.randombest import randombest_backbone_rmsd as _randombest_backbone_rmsd

def randombest_backbone_rmsd(
    refe, #str | np.ndarray,
    fraglib: np.ndarray,
    *,
    format: str,
    ntrajectories: int,
    max_rmsd: float,
    use_downstream_best: bool,
    random_seed = 0,
) -> tuple:
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

    Returns:
    - trajectories, a (poolsize, nresidues-fraglen+1) array of np.uint32
    - rmsds, a (poolsize) array of float
    """
    if isinstance(refe, str) and len(refe) < 200:
        raise ValueError("This function takes data content, not file names or URLs")

    main_state = _randombest_backbone_rmsd(
        refe, fraglib,
        format=format,
        ntrajectories=ntrajectories,
        max_rmsd=max_rmsd,
        use_downstream_best=use_downstream_best,
        random_seed=random_seed
    )
    natoms = main_state.refe.nfrags * main_state.refe.fraglen * len(main_state.refe.bb_atoms)
    last_stage = main_state.stages[-1]
    trajectories = last_stage.trajectories[:ntrajectories]
    scores = last_stage.scores[:ntrajectories]
    rmsds = np.sqrt(scores/natoms)
    return trajectories, rmsds