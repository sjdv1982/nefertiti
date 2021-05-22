import numpy as np
from nefertiti.procedures.randombest import randombest_backbone_rmsd

import logging
logging.basicConfig()
logging.getLogger("nefertiti").setLevel(logging.INFO)

fraglib = np.load("../fraglib/dummy.npy")
refe = open("1AVXA-unbound-heavy.pdb").read()
#refe = "\n".join(refe.splitlines()[:99])
main_state = randombest_backbone_rmsd(
    refe, fraglib,
    format="pdb",
    ntrajectories=10000,
    minblocksize=10000,
    use_downstream_best=False,
    max_rmsd=None
)
natoms = main_state.refe.nfrags * main_state.refe.fraglen * len(main_state.refe.bb_atoms)
threshold = main_state.stages[-1].scores[99]
max_rmsd = np.sqrt(threshold/natoms)
print(max_rmsd)

ntraj=1000
main_state = randombest_backbone_rmsd(
    refe, fraglib,
    format="pdb",
    ntrajectories=ntraj,
    use_downstream_best=False,
    max_rmsd=max_rmsd,
)

print(max_rmsd)
rmsds = np.sqrt(main_state.stages[-1].scores[:ntraj]/natoms)
print(rmsds.min(), rmsds.max())
