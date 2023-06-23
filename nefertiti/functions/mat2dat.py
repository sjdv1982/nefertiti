import numpy as np
from io import StringIO
from typing import Optional

from nefertiti.functions.rotmat2euler import rotmat2euler
    
def mat2dat(mat: np.ndarray, *, energies:Optional[np.ndarray]=None, conformers:Optional[np.ndarray]=None):
    """writes out 4x4 rotation-translation matrices in ATTRACT 2-body .dat format.
    
Energies and ligand conformer indices may be supplied as well.
The array of conformer indices must count from zero, although they are written out +1.

NOTE: the pivot for both receptor and ligand is set to the origin.
If the center-of-mass of the molecule is not in the origin, make sure that this is what is expected.
    """

    assert np.issubdtype(mat.dtype, np.number), mat.dtype
    assert mat.ndim == 3, mat.shape
    assert mat.shape[-2:] == (4, 4), mat.shape

    if energies is not None:
        assert np.issubdtype(energies.dtype, np.number), energies.dtype
        assert energies.ndim == 1, energies.shape
        assert len(energies) == len(mat), (len(energies), len(mat))

    if conformers is not None:
        assert np.issubdtype(conformers.dtype, np.number), conformers.dtype
        assert conformers.ndim == 1, conformers.shape
        assert len(conformers) == len(mat), (len(conformers), len(mat))

    rotmat = mat[:, :3, :3]
    euler = rotmat2euler(rotmat)
    translations = mat[:, 3, :3]

    result = StringIO()
    print("""#pivot 1 0 0 0
#pivot 2 0 0 0
#centered receptor: false
#centered ligands: false""", file=result)
    for n, ((phi, ssi, rot), (x, y, z)) in enumerate(zip(euler, translations)):
        print("#{}".format(n+1), file=result)
        if energies is not None:
            print("## Energy: {:.3f}".format(energies[n]), file=result)
        print("0 0 0 0 0 0", file=result)
        confstr = ""
        if conformers is not None:
            confstr = "{:d} ".format(conformers[n] + 1)
        print("{}{:.9f} {:.9f} {:.9f} {:.4f} {:.4f} {:.4f}".format(confstr, phi, ssi, rot, x, y, z), file=result)
    return result.getvalue()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("matfile", help="Input file (.npy) for 4x4 matrices")
    p.add_argument("datfile", help="ATTRACT .dat output file")
    p.add_argument("--energies", help="Extra input file (.npy) for energies")
    p.add_argument("--conformers", help="Extra input file (.npy) for conformers")

    args = p.parse_args()

    mat = np.load(args.matfile)
    
    energies = None
    if args.energies is not None:
        energies = np.load(args.energies)

    conformers = None
    if args.conformers is not None:
        conformers = np.load(args.conformers)

    dat_txt = mat2dat(mat, energies=energies, conformers=conformers)
    with open(args.datfile, "w") as f:
        f.write(dat_txt)
