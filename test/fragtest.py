import numpy as np
fraglib0 = np.load("../fraglib/dummy.npy")
print(fraglib0.shape)

from nefertiti.MainState import FragmentLibrary
from nefertiti.progressions.fraglib import load_backbone_fraglib
"""
from nefertiti.functions.fraglib import (
    prepare_fraglib_backbone, calc_fraglib_matrices_backbone
)
fraglib, residuals = prepare_fraglib_backbone(fraglib0)
matrices = calc_fraglib_matrices_backbone(fraglib)
"""
f = FragmentLibrary()
load_backbone_fraglib(f, fraglib0)

fraglib = f.coor.backbone4_centered
matrices = f.matrices

from nefertiti.functions.matrix import matmult, dotmat

traj = [0, 0, 0, 0, 0, 0]

coors = [] 
mat = np.eye(4)
fraglib2 = fraglib.reshape(len(fraglib), -1, 4)
for n in range(len(traj)):
    pos0 = traj[n]
    coor = dotmat(mat[None], fraglib2, [pos0])[:, :, :3]
    coors.append(coor[0])
    if n < len(traj) - 1:
        pos1 = traj[n+1]
        mats = matmult(mat[None], matrices[pos0, pos1][None], np.array([[0,0]]))
        mat = mats[0]

coors = np.concatenate(coors).reshape((len(coors), 4, 4, 3))
from nefertiti.functions.write_pdb import (
   build_pdb_fragment_backbone,  write_pdb
)
seq = "ACDEFGHIKLMNPQ"[:len(traj)+3]
pdb = build_pdb_fragment_backbone(coors,
    sequence=seq, 
    bb_atoms=["N", "CA", "C", "O"]
)
pdbdata = write_pdb(pdb)
with open("fragtest.pdb", "w") as f:
    f.write(pdbdata)
