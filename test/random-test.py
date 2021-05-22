import numpy as np
from nefertiti.protocols.randombest import randombest_backbone_rmsd

"""
import logging
logging.basicConfig()
logging.getLogger("nefertiti").setLevel(logging.INFO)
"""

fraglib = np.load("../fraglib/dummy.npy")
refe = np.load("../benchmarks/octacommon-aligned.npy")[0]
#refe = np.load("../benchmarks/dodecacommon-aligned.npy")[13]
_, rmsds = randombest_backbone_rmsd(
    refe, fraglib,
    format="npy",
    ntrajectories=200000,
    use_downstream_best=False,
    max_rmsd=None
)
threshold = rmsds[19] # top 0.01 %
print(threshold)

ntraj=1000
traj, rmsds = randombest_backbone_rmsd(
    refe, fraglib,
    format="npy",
    ntrajectories=ntraj,
    max_rmsd=threshold,
    use_downstream_best=True,
)
print(rmsds.min(), rmsds.max())