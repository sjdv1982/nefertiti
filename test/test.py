from nefertiti.MainState import StructureRepresentation
from nefertiti.progressions.prepare_backbone import prepare_backbone_from_pdb

import numpy as np

s = StructureRepresentation()
pdbdata = open("1AVXA-bound-heavy.pdb").read()
prepare_backbone_from_pdb(s, pdbdata)

s2 = StructureRepresentation()
pdbdata = open("1AVXA-unbound-heavy-rot.pdb").read()
prepare_backbone_from_pdb(s2, pdbdata)

from nefertiti.functions.superimpose import superimpose, superimpose_array

coor1 = s.coor_residue.backbone4_centered.reshape(-1, 4)[:, :3]
coor2 = s2.coor_residue.backbone4_centered.reshape(-1, 4)[:, :3]

#rotmat, rmsd = superimpose(coor2, coor1)
rotmat, rmsd = superimpose_array(coor2[None], coor1)

#coor2_unrot = coor2.dot(rotmat[0])
coor2_unrot = np.einsum("ijk,ikl->ijl", coor2[None], rotmat)[0]
coor2_unrot = coor2_unrot.reshape(-1, 4, 3)
from nefertiti.functions.write_pdb import (
    build_pdb_backbone, build_pdb_fragment_backbone,  write_pdb
)

atoms = build_pdb_backbone(
    #s.coor_residue.backbone4_centered,
    coor2_unrot,
    s.bb_atoms,
    s.sequence
)
"""
atoms2 = build_pdb_fragment_backbone(
    s.coor_fragment.backbone4_centered,
    s.bb_atoms,
    s.sequence
)
"""
print(write_pdb(atoms))


