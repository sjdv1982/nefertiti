from ..MainState import StructureRepresentation
from ..functions.prepare_backbone import prepare_backbone as prepare_backbone_func
from ..functions.parse_pdb import parse_pdb, get_backbone, get_xyz

import numpy as np

def prepare_backbone_from_pdb(s:StructureRepresentation, pdb) -> None:
    """Updates `s` by parsing `pdb` and calling the prepare_backbone progression
    `pdb` must be a file-like object or a text string in PDB format
    Also sets s.coor to the result of the parse_pdb function
    """
    if hasattr(pdb, "read"):
        pdbdata = pdb.read()
    else:
        pdbdata = pdb
    struc = parse_pdb(pdbdata)
    if s.bbatoms is None:
        s.bbatoms = ["N", "CA", "C", "O"]
    bb = get_backbone(struc, s.bbatoms)
    bb_coor = get_xyz(bb)
    s.coor = struc
    prepare_backbone(s, bb_coor)

def prepare_backbone(s: StructureRepresentation, bb_coor:np.ndarray) -> None:
    """Updates `s` by calling the prepare_backbone function"""
    if s.bbatoms is None:
        s.bbatoms = ["N", "CA", "C", "O"]
    if s.fraglen is None:
        s.fraglen = 4
    s.nresidues = len(bb_coor)
    s.coor_residue = {}
    s.coor_residue.backbone =  bb_coor
    bb_coor4, bb_coor_frag, bb_residuals_frag = prepare_backbone_func(
        bb_coor, s.fraglen, len(s.bbatoms)
    )
    s.coor_residue.backbone4 = bb_coor4 
    s.coor_fragment = {"nfrags": len(bb_coor_frag)}
    s.coor_fragment.backbone = bb_coor_frag[:, :, :, :3]
    s.coor_fragment.backbone4_centered = bb_coor_frag
    s.coor_fragment.backbone_residuals = bb_residuals_frag