from nefertiti.MainState import StructureRepresentation
from nefertiti.progressions.prepare_backbone import prepare_backbone_from_pdb


s = StructureRepresentation()
pdbdata = open("1AVXA-bound-heavy.pdb").read()
prepare_backbone_from_pdb(s, pdbdata)
