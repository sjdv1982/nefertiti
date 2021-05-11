import numpy as np
from nefertiti.protocols.greedy import greedy_backbone_rmsd

fraglib = np.load("../fraglib/dummy.npy")
refe = open("1AVXA-unbound-heavy.pdb").read()
main_state = greedy_backbone_rmsd(
    refe, fraglib,
    format="pdb",
    poolsize=1000
)
