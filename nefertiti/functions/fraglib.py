""" Functions related to fragment libraries

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

import numpy as np

def prepare_fraglib_backbone(fraglib: np.ndarray) -> np.ndarray:
    """Prepares a backbone fragment library
    ( array of shape (nfrag, fraglen,bbsize, 3) )
    - converts it to x,y,z,w form
    - centers each fragment
    Returns:
    - Centered fragments, shape (nfrag, fraglen, bbsize, 4)
    - Residuals (sum of squares) of each fragment
    """
    assert fraglib.ndim == 4, fraglib.shape
    assert fraglib.shape[-1] == 3, fraglib.shape
    fraglib_com = fraglib.reshape(len(fraglib), -1, 3).mean(axis=1)
    fraglib = fraglib - fraglib_com[:, None, None, :]
    fraglib4 = np.ones(fraglib.shape[:-1] + (4,))
    fraglib4[:, :, :, :3] = fraglib
    residuals = np.einsum("ijkl,ijkl->i", fraglib, fraglib)
    return fraglib4, residuals

def calc_fraglib_matrices(
    fraglib: np.ndarray,
    pre_indices: np.ndarray,
    post_indices: np.ndarray
):
    """Calculate fragment-fragment superposition matrices
    for a fragment library ( array of shape (nfrag, natoms, 3-or-4) )

    Returns an array M of (nfrag x nfrag) 4x4 matrices,
    (i.e. shape (nfrag, nfrag, 4, 4))
    where each 4x4 matrix M[i,j] contains the superposition 
      of fragment i and fragment j,
    where fragment i precedes fragment j in a trajectory of fragments

    pre_indices is an array of integers to define 
    which atoms to select from fragment i

    post_indices is the same for fragment j

    For each matrix MM in M,
    MM[:3, :3] is the rotation, MM[3, :3] is the translation,
    and MM[3,3]=1
    
    The matrices can be applied to fragment coordinates using
    nefertiti.functions.matrix.dotmat
    """
    assert fraglib.ndim == 3, fraglib.shape
    assert fraglib.shape[-1] in (3,4), fraglib.shape
    assert pre_indices.ndim == 1
    assert post_indices.ndim == 1

    from .superimpose import superimpose_array

    fraglib3 = fraglib[:, :, :3]
    first = fraglib3[:, pre_indices]
    first_com = first.mean(axis=1)
    first = first - first_com[:, None, :]
    matrices = np.zeros((len(fraglib),len(fraglib), 4, 4))
    matrices[:, :, 3, 3] = 1
    for n in range(len(fraglib)):
        last = fraglib3[n, post_indices]
        last_com = last.mean(axis=0)
        last = last - last_com
        curr_rotmats, _ = superimpose_array(first, last)
        matrices[n, :, :3, :3] = curr_rotmats
        matrices[n, :, 3, :3] = last_com[None] - first_com
    return matrices    

def calc_fraglib_matrices_backbone(
    fraglib: np.ndarray,
):
    """Calculates fragment-fragment superposition matrices
    for a backbone fragment library, 
    an array of shape (nfrag, nresidues, nbbatoms, 3-or-4)

    See calc_fraglib_matrices for more details
    """
    assert fraglib.ndim == 4, fraglib.shape
    assert fraglib.shape[-1] in (3,4), fraglib.shape

    natoms = fraglib.shape[1] * fraglib.shape[2]
    pre_indices = np.arange(natoms - fraglib.shape[2])
    post_indices = np.arange(fraglib.shape[2], natoms)
    return calc_fraglib_matrices(
        fraglib.reshape(len(fraglib), natoms, -1),
        pre_indices,
        post_indices
    )