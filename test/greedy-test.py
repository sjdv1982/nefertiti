import numpy as np
from nefertiti.protocols.greedy import greedy_backbone_rmsd

import logging
logging.basicConfig()
logging.getLogger("nefertiti").setLevel(logging.INFO)

fraglib = np.load("../fraglib/dummy.npy")
refe = open("1AVXA-unbound-heavy.pdb").read()
#refe = "\n".join(refe.splitlines()[:99])
main_state = greedy_backbone_rmsd(
    refe, fraglib,
    format="pdb",
    poolsize=200
)
natoms = main_state.refe.nfrags * main_state.refe.fraglen * len(main_state.refe.bb_atoms)
scores = main_state.stages[-1].scores[0]
rmsd = np.sqrt(scores/natoms)
print(rmsd)

# poolsize 100: 55s, 1.49A
# poolsize 200: 111s, 1.44702A
# poolsize 500: 277s, 1.44097A