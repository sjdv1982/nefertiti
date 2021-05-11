import numpy as np
fraglib = np.load("../fraglib/dummy.npy")
print(fraglib.shape)
fraglib_com = fraglib.reshape(len(fraglib), -1, 3).mean(axis=1)
fraglib = fraglib - fraglib_com[:, None, None, :]
fraglib4 = np.ones(fraglib.shape[:-1] + (4,))
fraglib4[:, :, :, :3] = fraglib
fraglib4c = fraglib4.reshape(len(fraglib4), -1, 4)

from nefertiti.functions.superimpose import superimpose_array

first = fraglib[:, :3].reshape(len(fraglib), -1, 3)
first_com = first.mean(axis=1)
first = first - first_com[:, None, :]
matrices = np.zeros((len(fraglib),len(fraglib), 4, 4))
matrices[:, :, 3, 3] = 1
for n in range(len(fraglib)):
    last = fraglib[n][1:].reshape(-1, 3)
    last_com = last.mean(axis=0)
    last = last - last_com
    curr_rotmats, _ = superimpose_array(first, last)
    matrices[n, :, :3, :3] = curr_rotmats
    matrices[n, :, 3, :3] = last_com[None] - first_com
    
from nefertiti.functions.matrix import matmult, dotmat

traj = [0, 0, 0, 0, 0, 0]

coors = [] 
mat = np.eye(4)
for n in range(len(traj)):
    pos0 = traj[n]
    coor = dotmat(mat[None], fraglib4c, [pos0])
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
