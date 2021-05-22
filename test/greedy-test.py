import sys
import numpy as np
from nefertiti.protocols.greedy import greedy_backbone_rmsd

import logging
logging.basicConfig()
logging.getLogger("nefertiti").setLevel(logging.INFO)

fraglib = np.load("../fraglib/dummy.npy")
refe = open("1AVXA-unbound-heavy.pdb").read()
trajectories, rmsds = greedy_backbone_rmsd(
    refe, fraglib,
    format="pdb",
    poolsize=int(sys.argv[1])
)
print(rmsds[:10])

# poolsize 100: 55s, 1.49418A
# poolsize 200: 111s, 1.44702A
# poolsize 500: 277s, 1.44097A