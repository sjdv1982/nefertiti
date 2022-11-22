import numpy as np
from math import pi

def rotmat2euler(rotmat: np.ndarray):
    """Converts Nx3x3 rotation matrices into Euler angles used in ATTRACT.
    This file was taken and adapted from the ATTRACT project (http://github.com/sjdv1982/attract),
      see copyright there.
    License: GPL
    """
    assert rotmat.ndim == 3, rotmat.shape
    assert rotmat.shape[-2:] == (3,3), rotmat.shape
    
    # In Nefertiti, rotation matrices are used as "matrix.dot(coordinates)""
    # However, the row/column convention for rotation matrices is opposite in ATTRACT
    rotmat = np.swapaxes(rotmat, 1, 2)
    
    phi = np.arctan2(rotmat[:, 1, 2], rotmat[:, 0, 2])
    ssi = np.arccos(np.minimum(np.maximum(rotmat[:, 2, 2], -1), 1))
    rot = np.arctan2(-rotmat[:, 2, 1],-rotmat[:, 2, 0])

    mask1 = np.fabs(rotmat[:, 2, 2]) >= 0.99999 #gimbal lock
    phi[mask1] = 0
    mask2 = np.fabs(rotmat[:, 0, 0]) >= 0.99999

    mask3 = rotmat[:, 0, 0] < 0
    rot[mask1 & mask2 & mask3] = pi
    rot[mask1 & mask2 & ~mask3] = 0

    mask4 = rotmat[:, 2, 2] < 0
    ssi[mask1 & mask2 & mask4] = pi
    ssi[mask1 & mask2 & ~mask4] = 0

    mask5 = (mask1 & ~mask2 & mask4)
    ssi[mask5] = pi
    rot[mask5] = -np.arccos(-rotmat[mask5, 0, 0])

    mask6 = (mask1 & ~mask2 & ~mask4)
    ssi[mask6] = 0
    rot[mask6] = np.arccos(rotmat[mask6, 0, 0])

    mask7 = (rotmat[:, 0, 1] < 0)
    rot[mask1 & mask7] *= -1

    return np.stack((phi,ssi,rot), axis=1)

if __name__ == "__main__":
    import sys
    rotmat = np.load(sys.argv[1])
    outputfile = sys.argv[2]
    euler = rotmat2euler(rotmat)
    np.save(outputfile, euler)