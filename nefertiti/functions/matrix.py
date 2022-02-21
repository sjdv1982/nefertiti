import numpy as np

def matmult(
    curr_matrices: np.ndarray, 
    mult_matrices: np.ndarray, 
    indices: np.ndarray
) -> np.ndarray:
    """Matrix multiplication
    No verification of array shape is done!

    Inputs: 
    - curr_matrices (cm), of shape Nx4x4
    - mult_matrices (mm) of shape M1xM2x4x4
    - Indices (ind) of shape Kx3, 
        with ind[:,0] in range 0..N
        and ind[:, 1] range 0..M1
        and ind[:, 2] range 0..M2
    
    returns r of shape Kx4x4
    
    for each k in range(K),
      r[k] = mm[m1,m2].dot(cm[n]),
    where n = ind[k][0] 
     and m1 = ind[k][1]
     and m2 = ind[k][2]
    """
    cm = curr_matrices[indices[:, 0]]
    m1, m2 = indices[:, 1], indices[:, 2]
    mm = mult_matrices[m1, m2]
    result = np.einsum("ijk,ikl->ijl", mm, cm) #diagonally broadcasted form of mm.dot(cm)
    return result


def dotmat(
    matrices: np.ndarray, 
    vectors: np.ndarray, 
    vector_indices: np.ndarray
) -> np.ndarray:
    """Vector-matrix multiplication
    No verification of array shape is done!

    Inputs: 
    - matrices of shape Mx4x4
    - vectors of shape NxVx4
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
