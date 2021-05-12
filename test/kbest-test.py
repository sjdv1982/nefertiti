import numpy as np
from nefertiti.protocols.kbest import kbest_backbone_rmsd

import logging
logging.basicConfig()
logging.getLogger("nefertiti").setLevel(logging.INFO)

fraglib = np.load("../fraglib/dummy.npy")
#refe = np.load("../benchmarks/octacommon-aligned.npy")[0]
refe = np.load("../benchmarks/dodecacommon-aligned.npy")[0]
k = 1
#k = 10000   # TODO: something wrong
main_state = kbest_backbone_rmsd(
    refe, fraglib,
    format="npy",
    k=k
)
natoms = main_state.refe.nfrags * main_state.refe.fraglen * len(main_state.refe.bb_atoms)
traj = main_state.stages[-1].trajectories[:k]
np.save("kbest-test-traj.npy", traj)
scores = main_state.stages[-1].scores[:k]
rmsd = np.sqrt(scores/natoms)
np.save("kbest-test-rmsd.npy", rmsd)
print(rmsd[:30])
