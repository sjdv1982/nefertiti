import sys, os
import numpy as np
from nefertiti.functions.matrix import matmult, dotmat
from nefertiti.functions.superimpose import superimpose_array
from nefertiti.functions.prepare_backbone import prepare_backbone
from nefertiti.functions.fraglib import prepare_fraglib_backbone, calc_fraglib_matrices_backbone

def calc_rmsd(traj, refe_struc0, fraglib):
    nfrags = traj.shape[1]
        
    fraglen = fraglib.shape[1]

    _, refe_struc, _, _ = prepare_backbone(refe_struc0, fraglen) # (nfrags, fraglen, 4, 3)
    assert len(refe_struc) == nfrags, (len(refe_struc), nfrags)
    refe_struc = refe_struc.reshape(-1, 4)[:, :3]

    fraglib, _ = prepare_fraglib_backbone(fraglib)
    fraglib_coor = fraglib.reshape(-1, fraglen*4, 4)
    fraglib_matrices = calc_fraglib_matrices_backbone(fraglib)

    rmsd = np.zeros(len(traj))

    chunksize = 10000
    curr_matrices = np.empty((chunksize, 4, 4))
    coors = np.empty((chunksize, nfrags, fraglen*4, 3))
    matmult_indices = np.empty((chunksize, 3), dtype=np.uint)
    matmult_indices[:, 0] = np.arange(chunksize)
    for n in range(0, len(traj), chunksize):
        n2 = min(n+chunksize, len(traj))
        real_chunksize = n2 - n
        print("{}/{}".format(n, len(traj)))
        curr_matrices[:] = np.eye(4)
        for i in range(nfrags):
            curr_coors = coors[:, i]
            curr_coors[:real_chunksize] = dotmat(
                curr_matrices[:real_chunksize],
                fraglib_coor,
                traj[n:n2, i]
            )
            if i + 1 < nfrags:
                matmult_indices[:real_chunksize, 1] = traj[n:n2, i]
                matmult_indices[:real_chunksize, 2] = traj[n:n2, i+1]
                curr_matrices[:real_chunksize] = matmult(
                    curr_matrices[:real_chunksize],
                    fraglib_matrices,
                    matmult_indices[:real_chunksize]
                )
        _, chunk_rmsds = superimpose_array(
            coors.reshape(len(coors), -1, 3)[:real_chunksize], 
            refe_struc
        )
        rmsd[n:n2] = chunk_rmsds
    
    return rmsd        


if __name__ == "__main__":
    trajfile = sys.argv[1]  # e.g. ../nnfit-suppmat/data/kbest-octa7-k-100000-traj.npy"
    refe_array = sys.argv[2]  # e.g. ../benchmarks/octacommon-aligned.npy
    refe_index = int(sys.argv[3])  # position in refe_array, starting at 1. "7" for octa7
    fraglib = sys.argv[4] # e.g. ../fraglib/dummy.npy
    resultfile = sys.argv[5]  # e.g. rmsd.npy
    
    assert os.path.exists(trajfile), trajfile
    refe_struc0 = np.load(refe_array)[refe_index-1]  # (nfrags, 4, 3) backbone coordinates
    traj = np.load(trajfile)
    fraglib = np.load(fraglib)
    rmsd = calc_rmsd(traj, refe_struc0, fraglib)
    np.save(resultfile, rmsd)