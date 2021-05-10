from nefertiti.MainState import StructureRepresentation
from nefertiti.functions.parse_pdb import parse_pdb, get_backbone, get_xyz
from nefertiti.functions.prepare_reference import prepare_reference_backbone


s = StructureRepresentation()
struc = parse_pdb(open("1AVXA-bound-heavy.pdb").read())
if s.bbatoms is None:
    s.bbatoms = ["N", "CA", "C", "O"]
bb = get_backbone(struc, s.bbatoms)
bb_coor = get_xyz(bb)
s.coor = struc

if s.bbatoms is None:
    s.bbatoms = ["N", "CA", "C", "O"]
s.fraglen = 4
s.nresidues = len(bb_coor)
s.coor_residue = {}
s.coor_residue.backbone =  bb_coor
bb_coor4, bb_coor_frag, bb_residuals_frag = prepare_reference_backbone(
    bb_coor, s.fraglen, len(s.bbatoms)
)
s.coor_residue.backbone4 = bb_coor4 
s.coor_fragment = {"nfrags": len(bb_coor_frag)}
s.coor_fragment.backbone = bb_coor_frag[:, :, :, :3]
s.coor_fragment.backbone4_centered = bb_coor_frag
s.coor_fragment.backbone_residuals = bb_residuals_frag