import numpy as np
from nefertiti.protocols.randombest import randombest_backbone_rmsd

import logging
logging.basicConfig()
logging.getLogger("nefertiti").setLevel(logging.INFO)

# unlike random-test.py, this is purely brute force! (use_downstream_best=False)
# 11m42s for full trypsin, only top 1 % (top 0.01 % for random-test.py)

fraglib = np.load("../fraglib/dummy.npy")
refe = open("1AVXA-unbound-heavy.pdb").read()
#refe = "\n".join(refe.splitlines()[:99])
_, rmsds = randombest_backbone_rmsd(
    refe, fraglib,
    format="pdb",
    ntrajectories=10000,
    use_downstream_best=False,
    max_rmsd=None
)
threshold = rmsds[99] 
print(threshold)

ntraj=1000
traj, rmsds = randombest_backbone_rmsd(
    refe, fraglib,
    format="pdb",
    ntrajectories=ntraj,
    use_downstream_best=False,
    max_rmsd=threshold,
)

print(threshold)
print(rmsds.min(), rmsds.max())
