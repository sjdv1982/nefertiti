from nefertiti.MainState import StructureRepresentation
from nefertiti.progressions.prepare_backbone import prepare_backbone_from_pdb

s = StructureRepresentation()
pdbdata = open("1AVXA-bound-heavy.pdb").read()
prepare_backbone_from_pdb(s, pdbdata)

from nefertiti.functions.write_pdb import write_pdb_backbone

print(s.coor_residue.backbone4_centered.shape)
print(len(s.sequence))
print(write_pdb_backbone(
    s.coor_residue.backbone4_centered,
    s.bb_atoms,
    s.sequence
))