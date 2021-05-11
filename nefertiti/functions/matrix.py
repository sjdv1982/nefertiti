import numpy as np

def matmult(curr_matrices, mult_matrices, indices):
    """Matrix multiplication
    No verification of array shape is done!

    Inputs: 
    - curr_matrices (cm), of shape Nx4x4
    - mult_matrices (mm) of shape Mx4x4
    - Indices (ind) of shape Kx2, 
        with ind[:,0] in range 0..N
        and ind[:, 1] range 0..M
    
    returns r of shape Kx4x4
    
    for each k in range(K),
      r[k] = mm[m].dot(cm[n]),
    where n = ind[k][0] and m = ind[k][1]
    """
    cm = curr_matrices[indices[:, 0]]
    mm = mult_matrices[indices[:, 1]]
    result = np.einsum("ijk,ikl->ijl", mm, cm) #diagonally broadcasted form of mm.dot(cm)
    return result


def dotmat(matrices, vectors, vector_indices):
    """Vector-matrix multiplication
    No verification of array shape is done!

    Inputs: 
    - vectors of shape NxVx4
    - matrices of shape Mx4x4
    - Indices (ind) of shape M, 
        with ind in range 0..N
    
    returns r of shape MxVx3
    
    for each m in range(M),
      r[m] = vector[ind[m]].dot(matrix[m])
    and then the 4-dimensional vector x,y,z,w has dimension w removed
    """
    vv = vectors[vector_indices]
    result = np.einsum("ijk,ikl->ijl", vv, matrices) #diagonally broadcasted form of vv.dot(matrices)
    return result[:, :, :3]
