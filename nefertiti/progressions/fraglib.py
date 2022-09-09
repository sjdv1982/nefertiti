""" (description)

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""
import numpy as np

from ..MainState import FragmentCoordinateRepresentation, FragmentLibrary

def load_backbone_fraglib(
    f: FragmentLibrary, 
    fraglib: np.ndarray
) -> None:
    """Updates `f` by calling the functions 
    `prepare_fraglib_backbone` and  `calc_fraglib_matrices_backbone`
    and setting:
    - f.matrices
    - f.coor.backbone4_centered
    - f.coor.backbone_residuals

    f.nfrags, f.fraglen and f.bb_atoms are set as well, if not defined.
    """
    from ..functions.fraglib import (
        prepare_fraglib_backbone, calc_fraglib_matrices_backbone
    )
    if f.nfrags is not None:
        assert f.nfrags == fraglib.shape[0], (f.nfrags, fraglib.shape)
    else:
        f.nfrags = fraglib.shape[0]
    
    if f.fraglen is not None:
        assert f.fraglen == fraglib.shape[1], (f.fraglen, fraglib.shape)
    else:
        f.fraglen = fraglib.shape[1]
    
    if f.bb_atoms is None:
        assert fraglib.shape[2] == 4, fraglib.shape
        f.bb_atoms = ["N", "CA", "C", "O"]
    else:
        assert len(f.bb_atoms) == fraglib.shape[2], (f.bb_atoms, fraglib.shape)

    fraglib, residuals = prepare_fraglib_backbone(fraglib)
    matrices = calc_fraglib_matrices_backbone(fraglib)
    f.matrices = matrices
    if f.coor is None:
        f.coor = FragmentCoordinateRepresentation()
    f.coor.backbone4_centered = fraglib
    f.coor.backbone_residuals = residuals