import numpy as np
import os
from nefertiti.functions.parse_pdb import parse_pdb, get_backbone, get_xyz
from nefertiti.functions.align_common_frame import align_common_frame
from nefertiti.functions.write_pdb import write_multi_pdb, build_pdb_backbone
from nefertiti.functions.superimpose import superimpose

fraglibdir = "cutoff1.9-freq0.025-ward"
assert os.path.exists(fraglibdir)

REDUNDANCY_THRESHOLD=1.0  # 1 A cutoff

letters = []
for l in open(os.path.join(fraglibdir, "letters-ncls.txt")):
    ll = l.split()
    letters.append([ll[0], int(ll[1])])

def fix_pdbdata(pdbdata):
    # Remove alternative residue column
    result = []
    col = 16
    for l in pdbdata.splitlines():
        if l.startswith("ATOM "):
            l = l[:col] + " " + l[col+1:]
        result.append(l)
    return "\n".join(result)

# 1. Parse fragment library
#    a. fix PDB files
#    b. parse PDB with Biopython
#    c. extract backbone coordinates
#    d. align coordinates in common reference frame (not strictly necessary)
bb_atoms = ["N", "CA", "C", "O"]
fragments_coor = []
for letter, nproto in letters:
    for n in range(nproto):
        pdbfile = os.path.join(fraglibdir, "{}.fittmpl.proto{}".format(letter, n))
        print(pdbfile)
        pdbdata = open(pdbfile).read()
        pdbdata = fix_pdbdata(pdbdata)
        pdb = parse_pdb(pdbdata)
        backbone_coor = get_xyz(get_backbone(pdb, bb_atoms))
        assert backbone_coor.shape == (4,4,3)
        aligned_backbone_coor = align_common_frame(backbone_coor)
        fragments_coor.append(aligned_backbone_coor)

# 2. Write out fragment library as single PDB file
fragments = [build_pdb_backbone(fragments_coor[n], bb_atoms) for n in range(len(fragments_coor))]
fraglib = np.stack(fragments)
fraglib_pdb = write_multi_pdb(fraglib)
with open("fraglib-pepfold-original.pdb", "w") as f:
    f.write(fraglib_pdb)

# 3. Remove redundancy
#  a. Calculate pairwise RMSD matrix
#  b. Cluster the fragments
nfrag = len(fragments)
rmsd_mat = np.zeros((nfrag, nfrag))
for n in range(nfrag):
    print("{}/{}".format(n+1, nfrag))
    fragcoor1 = fragments_coor[n].reshape(-1, 3)
    for nn in range(n+1, nfrag):
        fragcoor2 = fragments_coor[nn].reshape(-1, 3)
        _, rmsd = superimpose(fragcoor1, fragcoor2)
        rmsd_mat[n, nn] = rmsd
        rmsd_mat[nn, n] = rmsd

redundancy_mat = (rmsd_mat < REDUNDANCY_THRESHOLD)

clustered = 0
clusters = []
while clustered < nfrag:
    neigh = redundancy_mat.sum(axis=0)
    heart = neigh.argmax()
    cluster = np.where(redundancy_mat[heart])[0]
    for cs in cluster:
        redundancy_mat[cs,:] = False
        redundancy_mat[:, cs] = False
    cluster = [heart+1] + [v+1 for v in cluster if v != heart]
    clusters.append(cluster)
    clustered += len(cluster)

clusters.sort()
with open("fraglib-clustering.out", "w") as f:
    print("Clustering:")
    for n in range(len(clusters)):
        print(n+1, clusters[n])
        print(n+1, clusters[n], file=f)

# 3. Write out fragment library:
#  - as Numpy array of backbone coordinates (to be used in calculations)
#  - as single PDB file (for visualization)
fragments_coor_nonredundant = [fragments_coor[cluster[0]-1] for cluster in clusters]
fraglib_coor_nonredundant = np.stack(fragments_coor_nonredundant)
np.save("fraglib-pepfold.npy", fraglib_coor_nonredundant)

fragments_nonredundant = [fragments[cluster[0]-1] for cluster in clusters]
fraglib_nonredundant = np.stack(fragments_nonredundant)
fraglib_pdb_nonredundant = write_multi_pdb(fraglib_nonredundant)
with open("fraglib-pepfold.pdb", "w") as f:
    f.write(fraglib_pdb_nonredundant)
