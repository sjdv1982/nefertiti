import numpy as np
from nefertiti.protocols.randombest import randombest_backbone_rmsd

import logging
logging.basicConfig()
logging.getLogger("nefertiti").setLevel(logging.INFO)

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
