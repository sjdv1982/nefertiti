import numpy as np

from .nefertiti.MainState import FragmentLibrary
from .nefertiti.progressions.fraglib import load_backbone_fraglib

f = FragmentLibrary()
load_backbone_fraglib(f, fraglib0)

fraglib = f.coor.backbone4_centered
matrices = f.matrices

from .nefertiti.functions.matrix import matmult, dotmat

coors = [] 
mat = np.eye(4)
fraglib2 = fraglib.reshape(len(fraglib), -1, 4)
for n in range(len(traj)):
    pos0 = traj[n]
    coor = dotmat(mat[None], fraglib2, [pos0])[:, :, :3]
    coors.append(coor[0])
    if n < len(traj) - 1:
        pos1 = traj[n+1]
        mats = matmult(mat[None], matrices[pos0, pos1][None, None], np.array([[0,0,0]]))
        mat = mats[0]

coors = np.concatenate(coors).reshape((len(coors), 4, 4, 3))
from .nefertiti.functions.write_pdb import (
   build_pdb_fragment_backbone,  write_pdb
)
seq = "IVGGYTCAANSIPYQVSLNSGSHFCGGSLINSQWVVSAAHCYKSRIQVRLGEHNIDVLEGNEQFINAAKIITHPNFNGNTLDNDIMLIKLSSPATLNSRVATVSLPRSCAAAGTECLISGWGNTKSSGSSYPSLLQCLKAPVLSDSSCKSSYPGQITGNMICVGFLEGGKDSCQGDSGGPVVCNGQLQGIVSWGYGCAQKNKPGVYTKVCNYVNWIQQTIAAN"
seq = seq[:len(traj)+3]
pdb = build_pdb_fragment_backbone(coors,
    sequence=seq, 
    bb_atoms=["N", "CA", "C", "O"]
)
pdbdata = write_pdb(pdb)

result = pdbdata