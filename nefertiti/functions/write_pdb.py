"""Write binary format back to PDB file

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""
import numpy as np
from typing import List

code = {
  "ALA": "A",
  "CYS": "C",
  "ASP": "D",
  "GLU": "E",
  "PHE": "F",
  "GLY": "G",
  "HIS": "H",
  "ILE": "I",
  "LYS": "K",
  "LEU": "L",
  "MET": "M",
  "ASN": "N",
  "PRO": "P",
  "GLN": "Q",
  "ARG": "R",
  "SER": "S",
  "THR": "T",
  "VAL": "V",
  "TRP": "W",
  "TYR": "Y",
  "UNK": "X",
}
code_rev = {v:k for k,v in code.items()}

# adapted from Bio.PDB.PDBIO.py

_ATOM_FORMAT_STRING = (
    "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s\n"
)
def write_pdb_atom(atom) -> str:
    name = atom["name"].decode()
    if not name.startswith("H"):
        name = " " + name
    occ = atom["occupancy"]
    if occ > 100:
        occ = 100
    if occ < 0:
        occ = 0
    args = (
        "ATOM  " if atom["hetero"].decode() == " " else "HETATM",
        atom["index"],
        name,
        atom["altloc"].decode(),
        atom["resname"].decode(),
        atom["chain"].decode(),
        atom["resid"],
        atom["icode"].decode(),
        atom["x"],
        atom["y"],
        atom["z"],
        occ,
        atom["bfactor"],
        atom["segid"].decode(),
        atom["element"].decode(),
        "",
    )
    return _ATOM_FORMAT_STRING % args
#/adapted

def write_pdb(struc: np.ndarray) -> str:
    from . import parse_pdb
    assert struc.dtype == parse_pdb.atomic_dtype
    assert struc.ndim == 1
    pdb = ""
    for atom in struc:
        line = write_pdb_atom(atom)
        pdb += line
    return pdb

def build_pdb_backbone(
    struc: np.ndarray, 
    bb_atoms: List[str],
    sequence:str = None,
    *,
    atomindex_offset=0,
    resid_offset=0,
) -> np.ndarray:
    assert struc.ndim == 3, struc.shape
    assert struc.shape[1] == len(bb_atoms), struc.shape
    assert struc.shape[2] in (3,4), struc.shape
    assert len(struc) < 10000
    if sequence is not None:
        assert len(sequence) == len(struc)
    from . import parse_pdb
    newstruc = np.empty(
        (len(struc), len(bb_atoms)),
        parse_pdb.atomic_dtype
    )
    for resnr in range(len(struc)):
        res = newstruc[resnr]
        for anr in range(len(bb_atoms)):
            s = struc[resnr][anr]
            a = newstruc[resnr][anr]
            a["model"] = 1
            a["hetero"] = ""
            a["name"] = bb_atoms[anr].encode()
            a["altloc"] = " "
            resname = code_rev[sequence[resnr]] if sequence is not None else "ALA"
            a["resname"] = resname.encode()
            a["chain"] = " "
            a["index"] = len(bb_atoms) * resnr + anr + atomindex_offset + 1
            a["icode"] = " "
            a["resid"] = resnr + resid_offset + 1
            a["x"] = s[0]
            a["y"] = s[1]
            a["z"] = s[2]
            a["occupancy"] = 1
            a["segid"] = ""
            a["element"] = bb_atoms[anr][0].encode()
    return newstruc.flatten()

def write_pdb_backbone(
    struc: np.ndarray, 
    bb_atoms: List[str],
    sequence:str = None,
) -> str:
    atoms = build_pdb_backbone(struc, bb_atoms, sequence)
    pdb = write_pdb(atoms)
    return pdb