import numpy as np

def euler2rotmat(euler):
    """Converts Euler angles used in ATTRACT into Nx3x3 rotation matrices.
    This file was taken and adapted from the ATTRACT project (http://github.com/sjdv1982/attract),
      see copyright there.
    License: GPL
    """

    assert euler.ndim == 2
    assert euler.shape[1] == 3
    phi, ssi, rot = euler[:, 0], euler[:, 1], euler[:, 2]
    assert phi.shape == ssi.shape == rot.shape
    cs = np.cos(ssi)
    cp = np.cos(phi)
    ss = np.sin(ssi)
    sp = np.sin(phi)
    cscp = cs*cp
    cssp = cs*sp
    sscp = ss*cp
    sssp = ss*sp
    crot = np.cos(rot)
    srot = np.sin(rot)

    r1 = crot * cscp + srot * sp
    r2 = srot * cscp - crot * sp
    r3 = sscp

    r4 = crot * cssp - srot * cp
    r5 = srot * cssp + crot * cp
    r6 = sssp

    r7 = -crot * ss
    r8 = -srot * ss
    r9 = cs
    result = np.zeros(phi.shape + (3,3))
    result[:, 0, 0] = r1
    result[:, 0, 1] = r2
    result[:, 0, 2] = r3
    result[:, 1, 0] = r4
    result[:, 1, 1] = r5
    result[:, 1, 2] = r6
    result[:, 2, 0] = r7
    result[:, 2, 1] = r8
    result[:, 2, 2] = r9

    # In Nefertiti, rotation matrices are used as "matrix.dot(coordinates)""
    # However, the row/column convention for rotation matrices is opposite in ATTRACT
    result = np.swapaxes(result, 1, 2)

    return result

if __name__ == "__main__":
    import sys
    euler = np.load(sys.argv[1])
    outputfile = sys.argv[2]
    rotmat = euler2rotmat(euler)
    np.save(outputfile, rotmat)