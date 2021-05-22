import numpy as np
from nefertiti.procedures.randombest import randombest_backbone_rmsd

"""
import logging
logging.basicConfig()
logging.getLogger("nefertiti").setLevel(logging.INFO)
"""

fraglib = np.load("../fraglib/dummy.npy")
refe = np.load("../benchmarks/octacommon-aligned.npy")[0]
#refe = np.load("../benchmarks/dodecacommon-aligned.npy")[13]
main_state = randombest_backbone_rmsd(
    refe, fraglib,
    format="npy",
    ntrajectories=200000,
    use_downstream_best=False,
    max_rmsd=None
)
natoms = main_state.refe.nfrags * main_state.refe.fraglen * len(main_state.refe.bb_atoms)
threshold = main_state.stages[-1].scores[19] # top 0.01 %
max_rmsd = np.sqrt(threshold/natoms)
print(max_rmsd)

ntraj=1000
main_state = randombest_backbone_rmsd(
    refe, fraglib,
    format="npy",
    ntrajectories=ntraj,
    max_rmsd=max_rmsd,
    use_downstream_best=True,
)
rmsds = np.sqrt(main_state.stages[-1].scores[:ntraj]/natoms)
print(rmsds.min(), rmsds.max())