import numpy as np

from nefertiti.functions.euler2rotmat import euler2rotmat
    
def read_dat(dat_txt: str) -> tuple:
    """Reads ATTRACT 2-body .dat format.
    Returns tuple of (mat, energies, conformers)
     where mat is an array of 4x4 rotation-translation matrices
    Energies and conformers are None if they aren't present in the file
    """
    npivots = 0
    ncentered = 0
    header_done = False
    lines = dat_txt.splitlines()
    for lnr, l in enumerate(lines):
        l = l.strip()
        if l.startswith("##"):
            continue
        if not l.startswith("#"):
            raise ValueError(l)
        l = l[1:].strip()
        if l.split() == ["pivot", "auto"]:
            if npivots:
                raise ValueError(l)
            npivots = 2
            continue
        if npivots < 2:
            ll = l.split()
            if ll[:1] != ["pivot"]:
                raise ValueError(l)
            if ll[1] != str(npivots+1):
                raise ValueError(l)
            pivot = [float(f) for f in ll[2:]]
            if pivot != [0.0,0.0,0.0]:
                raise ValueError("You must depivotize first")
            npivots += 1
            continue
        if npivots == 2:
            if ncentered == 0:
                if l != "centered receptor: false":
                    raise ValueError(l)
                ncentered += 1
                continue
            if ncentered == 1:
                if l != "centered ligands: false":
                    raise ValueError(l)
                ncentered += 1
                
                header_done = True
                pos = lnr + 1
                break
                
    if not header_done:
        raise ValueError("Cannot read ATTRACT .dat file (header)")

    if lines[pos] != "#1":
        raise ValueError(lines[pos])
    
    euler = None
    offset = None
    energies = None
    conformers = None
    rec_offset = None
    
    lnr = pos
    while lnr < len(lines) - 1:
        lnr += 1

        l = lines[lnr].strip()
        if l.startswith("###"):
            continue
        if l.startswith("##"):
            ll = l.split()
            if ll[1] == "Energy:":
                energies = []
            continue
        
        if l.startswith("##"):
            raise ValueError("Cannot read ATTRACT .dat file (first structure)")
        
        ll = l.split()
        dofs = [float(f) for f in ll]
        if len(dofs) != 6 or dofs[:3] != [0.0] * 3:
            raise ValueError("Receptor: " + l)
        rec_offset = dofs[3:]

        lnr += 1
        l = lines[lnr].strip()
        assert not l.startswith("##")
        ll = l.split()
        dofs = [float(f) for f in ll]
        if len(dofs) not in (6, 7):
            raise ValueError("Ligand: " + l)
        euler = []
        offset = []
        if len(dofs) == 7:
            conformers = []
        break

    if euler is None:
        raise ValueError("Cannot read ATTRACT .dat file (first structure)")

    lnr = pos
    while lnr < len(lines) - 1:
        lnr += 1

        l = lines[lnr].strip()
        if l.startswith("###"):
            continue
        if l.startswith("##"):
            ll = l.split()
            if ll[1] == "Energy:":
                if energies is None or len(energies) != len(euler):
                    raise ValueError("Cannot read ATTRACT .dat file (structure {})".format(len(euler)))                
                energies.append(float(ll[2]))
            continue
        
        if l.startswith("##"):
            raise ValueError("Cannot read ATTRACT .dat file (structure {})".format(len(euler)))

        ll = l.split()
        dofs = [float(f) for f in ll]
        
        if len(dofs) != 6 or dofs[:3] != [0.0] * 3 or dofs[3:] != rec_offset:
            raise ValueError("Cannot read ATTRACT .dat file (structure {})".format(len(euler)))
        
        lnr += 1
        l = lines[lnr].strip()
        assert not l.startswith("##")
        ll = l.split()
        dofs = [float(f) for f in ll]
        if conformers is None:
            if len(dofs) != 6:
                raise ValueError("Cannot read ATTRACT .dat file (structure {})".format(len(euler)))
        else:
            if len(dofs) != 7:
                raise ValueError("Cannot read ATTRACT .dat file (structure {})".format(len(euler)))
            conformers.append(int(dofs[0]))

        euler.append(dofs[-6:-3])
        offset.append(dofs[-3:])

        lnr += 1
        if lnr == len(lines):
            break
        if lines[lnr] != "#" + str(len(euler) + 1):
            raise ValueError("Cannot read ATTRACT .dat file (structure {})".format(len(euler)))

    assert len(euler) == len(offset)
    #print(len(euler), len(offset), (len(conformers) if conformers else None), (len(energies) if energies else None))
    rotmat = euler2rotmat(np.array(euler))
    mat = np.ones((len(rotmat), 4, 4))
    mat[:, :3, :3] = rotmat
    mat[:, 3, :3] = np.array(offset) - rec_offset
    return mat, energies, conformers

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("datfile", help="ATTRACT .dat file")
    p.add_argument("mat_outputfile", help="Output file (.npy) for 4x4 matrices")
    p.add_argument("--energies", help="Output file (.npy) for energies")
    p.add_argument("--conformers", help="Output file (.npy) for conformer indices, counting from one")

    args = p.parse_args()
    with open(args.datfile) as f:
        dat_txt = f.read()
    mat_outputfile = args.mat_outputfile

    mat, energies, conformers = read_dat(dat_txt)

    energies_outputfile = args.energies
    if energies_outputfile is not None:
        assert energies is not None
    conformers_outputfile = args.conformers
    if conformers_outputfile is not None:
        assert conformers is not None

    np.save(mat_outputfile, mat)

    if energies_outputfile is not None:    
        np.save(energies_outputfile, energies)
    if conformers_outputfile is not None:
        np.save(conformers_outputfile, conformers)
