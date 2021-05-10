""" (description)

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

from .State import *
from builtins import *

class MainState(State):
    _state = {
        "refe": "StructureRepresentation", # The reference structure
        "fraglib": "FragmentLibrary",
    }

################################################################
# Structure representation (of reference structure)
################################################################

class StructureRepresentation(State):
    _state = {
        "coor": ("ndarray", validate_coor_dtype),
        "coor_fragment": "FragmentCoordinateRepresentation",
        "coor_residue": "CoordinateRepresentation",
        "nresidues": "uint",
        "fraglen": "uint",
        "bbatoms": "ListOf(str)",
    }

class CenteredState(State):
    def _check_centered(self):
        for attr in self._state:
            if not attr.endswith("_centered"):
                continue
            value = getattr(self, attr)
            if value is None:
                continue
            value2 = value.reshape(-1, 4)[:, :3]
            com = value2.mean(axis=0)
            if not np.allclose(com, np.zeros(3)):
                raise ValueError("Not centered", attr, com)        
    def validate(self):
        self._check_centered()

class FragmentCoordinateRepresentation(CenteredState):
    """Note that different representations may not have the same center-of-mass!"""
    _state = {
        "nfrags": "uint",
        "backbone": ("ndarray", "(nfrags, fraglen, len(bbatoms), 3)"),
        "backbone4": ("ndarray", "(nfrags, fraglen, len(bbatoms), 4)"),
        "backbone4_centered": ("ndarray", "(nfrags, fraglen, len(bbatoms), 4)"),
        "backbone_residuals": ("ndarray", "(nfrags,)"),
        "ca": ("ndarray", "(nfrags, fraglen, 3)"),
        "ca4": ("ndarray", "(nfrags, fraglen, 4)"),
        "ca4_centered": ("ndarray", "(nfrags, fraglen, 4)"),
        "ca_residuals": ("ndarray", "(nfrags,)"),
    }

class CoordinateRepresentation(CenteredState):
    """Note that different representations may not have the same center-of-mass!"""
    _state = {
        "backbone": ("ndarray", "(nresidues, len(bbatoms), 3)"),
        "backbone4": ("ndarray", "(nresidues, len(bbatoms), 4)"),
        "backbone4_centered": ("ndarray", "(nresidues, len(bbatoms), 4)"),
        "backbone_residual": "float",
        "ca": ("ndarray", "(nresidues, 3)"),
        "ca4": ("ndarray", "(nresidues, 4)"),
        "ca4_centered": ("ndarray", "(nresidues, 4)"),
        "ca_residual": "float",
    }

################################################################
# Fragment library representation (of reference structure)
################################################################

class FragmentLibrary(CenteredState):
    _state = {
        "coor": "FragmentCoordinateRepresentation"
        
    }
# fraglib: coors and matrices