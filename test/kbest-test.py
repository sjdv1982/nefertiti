import numpy as np
from nefertiti.protocols.kbest import kbest_backbone_rmsd

import logging
logging.basicConfig()
logging.getLogger("nefertiti").setLevel(logging.INFO)

fraglib = np.load("../fraglib/dummy.npy")
#refe = np.load("../benchmarks/octacommon-aligned.npy")[0] #25 sec for k=10000, 2m38 for k=100000
#refe = np.load("../benchmarks/octacommon-aligned.npy")[3] #1m56 sec for k=10000, 8m5 for k=100000
#refe = np.load("../benchmarks/octacommon-aligned.npy")[16] #44 sec for k=10000, 4m5 for k=100000

#refe = np.load("../benchmarks/dodecacommon-aligned.npy")[0]
refe = np.load("../benchmarks/dodecacommon-aligned.npy")[13]
#k = 1   # 14s for [0], 12s for [13]
#k = 10000   # 4m40 for [0], 10m30s for [13]
k = 100000   # 28m08 for [0] 62m07s for [13]
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
