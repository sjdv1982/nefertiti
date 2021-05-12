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
    poolsize=1000
)
