""" (description)

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

from .State import *
from builtins import *

class MainState(State):
    _state = {
        "fraglen": "uint",
        "refe": "StructureRepresentation", # The reference structure
        "fraglib": "FragmentLibrary",
        "nfrags": "uint", # Here: number of fragments to describe the trajectory
        "bb_atoms": "ListOf(str)",
        "stages": "ListOf(Stage)",
    }

################################################################
# Structure representation (of reference structure or fragment library)
################################################################

def validate_sequence(seq: str,  state: "State"):
    for aa in seq:
        assert aa in "ACDEFGHIKLMNPQRSTVWXYZ", aa

class StructureRepresentation(State):
    _state = {
        "coor": ("ndarray", validate_coor_dtype),
        "sequence": ("str", validate_sequence),
        "coor_fragment": "FragmentCoordinateRepresentation",
        "coor_residue": "CoordinateRepresentation",
        "nresidues": "uint",
        "fraglen": "uint",
        "nfrags": "uint", # might be inherited from parent instead
        "bb_atoms": "ListOf(str)",
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
    def _validate(self):
        self._check_centered()

class FragmentCoordinateRepresentation(CenteredState):
    """Note that different representations may not have the same center-of-mass!"""
    _state = {
        # From parent:
        # "nfrags": "uint",   
        # "fraglen": "uint",
        # "bb_atoms": "ListOf(str)",

        "backbone": ("ndarray", "(nfrags, fraglen, len(bb_atoms), 3)"),
        "backbone4": ("ndarray", "(nfrags, fraglen, len(bb_atoms), 4)"),
        "backbone4_centered": ("ndarray", "(nfrags, fraglen, len(bb_atoms), 4)"),
        "backbone_residuals": ("ndarray", "(nfrags,)"),
        "backbone_com": ("ndarray", "(nfrags, 3)"),
        "ca": ("ndarray", "(nfrags, fraglen, 3)"),
        "ca4": ("ndarray", "(nfrags, fraglen, 4)"),
        "ca4_centered": ("ndarray", "(nfrags, fraglen, 4)"),
        "ca_residuals": ("ndarray", "(nfrags,)"),
        "ca_com": ("ndarray", "(nfrags, 3)"),
    }

class CoordinateRepresentation(CenteredState):
    """Note that different representations may not have the same center-of-mass!"""
    _state = {
        "backbone": ("ndarray", "(nresidues, len(bb_atoms), 3)"),
        "backbone4": ("ndarray", "(nresidues, len(bb_atoms), 4)"),
        "backbone4_centered": ("ndarray", "(nresidues, len(bb_atoms), 4)"),
        "backbone_residual": "float",
        "ca": ("ndarray", "(nresidues, 3)"),
        "ca4": ("ndarray", "(nresidues, 4)"),
        "ca4_centered": ("ndarray", "(nresidues, 4)"),
        "ca_residual": "float",
    }

################################################################
# Fragment library 
################################################################

class FragmentLibrary(CenteredState):
    _state = {
        "nfrags": "uint",  # Here, the size of the fragment library
        "fraglen": "uint",
        "bb_atoms": "ListOf(str)",
        "coor": "FragmentCoordinateRepresentation",
        "matrices":("ndarray", "(nfrags,nfrags,4,4)"),
    }

################################################################
# Stages of growing the trajectory 
################################################################

class Stage(State):
    _state = {
        "size": "uint",      # current number of trajectories that this stage holds
        "maxsize": "uint",  # maximum number of trajectories that this stage can hold

        "fragindex": "uint",  # Here: number of fragments at the current stage
        "trajectories": ("ndarray", "(-1,fragindex)"),  #16 bit unsigned int
        # unsigned integer; trajectories of all fragments until now

        "matrices": ("ndarray", "(-1,4,4)"),
        "scores": ("ndarray", "(-1,)"),  # lower means better
        "score_threshold": "float",  # reject all scores higher than threshold

        # If/as long as we need to store coordinates, rather than matrices
        "nfrags": "uint",  # Here, the same as maxsize
        "coor": "FragmentCoordinateRepresentation",  #inherits nfrags from here

        # If RMSD calculation to a reference
        "covar": ("ndarray", "(-1,3,3)"),
        "residuals": ("ndarray", "(-1,)"),
        "fragcoms": ("ndarray", "(-1, fragindex, 3)"),
        
        # If we need to store *all* matrices (e.g. for forcefield calculations)
        # We could also re-compute these at any time from the trajectories
        "all_matrices": ("ndarray", "(-1,fragindex,4,4)"),
    }
