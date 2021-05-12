""" (description)

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

from ..MainState import StructureRepresentation
from ..functions.prepare_backbone import prepare_backbone as prepare_backbone_func
from ..functions.parse_pdb import parse_pdb, get_backbone, get_xyz, get_sequence

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
    if s.bb_atoms is None:
        s.bb_atoms = ["N", "CA", "C", "O"]
    bb = get_backbone(struc, s.bb_atoms)
    bb_coor = get_xyz(bb)
    s.coor = struc
    s.sequence = get_sequence(struc)
    prepare_backbone(s, bb_coor)

def prepare_backbone(s: StructureRepresentation, bb_coor:np.ndarray) -> None:
    """Updates `s` by calling the prepare_backbone function"""
    if s.bb_atoms is None:
        s.bb_atoms = ["N", "CA", "C", "O"]
    if s.fraglen is None:
        s.fraglen = 4
    s.nresidues = len(bb_coor)
    if s.nfrags is None:
        s.nfrags = s.nresidues - s.fraglen + 1
    s.coor_residue = {}
    s.coor_residue.backbone =  bb_coor
    prep = prepare_backbone_func(
        bb_coor, s.fraglen, len(s.bb_atoms)
    )
    bb_coor4, bb_coor_frag, bb_residuals_frag, bb_com_frag = prep
    s.coor_residue.backbone4 = bb_coor4
    com = bb_coor4.reshape(-1, 4).mean(axis=0)
    com[3] = 0
    s.coor_residue.backbone4_centered = bb_coor4 - com
    s.coor_fragment = {}
    s.coor_fragment.backbone4_centered = bb_coor_frag
    s.coor_fragment.backbone_residuals = bb_residuals_frag
    s.coor_fragment.backbone_com = bb_com_frag

def select_last_backbone(s: StructureRepresentation, nfrags:int) -> None:
    """Updates `s` that has a prepared backbone, by selecting the last residues"""
    nresidues = int(nfrags + s.fraglen - 1)
    s.nresidues = nresidues
    s.nfrags = nfrags
    s.coor_residue.backbone =  s.coor_residue.backbone[-nresidues:]
    prep = prepare_backbone_func(
        s.coor_residue.backbone, s.fraglen, len(s.bb_atoms)
    )
    bb_coor4, bb_coor_frag, bb_residuals_frag, bb_com_frag = prep
    s.coor_residue.backbone4 = bb_coor4
    com = bb_coor4.reshape(-1, 4).mean(axis=0)
    com[3] = 0
    s.coor_residue.backbone4_centered = bb_coor4 - com
    s.coor_fragment = {}
    s.coor_fragment.backbone4_centered = bb_coor_frag
    s.coor_fragment.backbone_residuals = bb_residuals_frag
    s.coor_fragment.backbone_com = bb_com_frag    