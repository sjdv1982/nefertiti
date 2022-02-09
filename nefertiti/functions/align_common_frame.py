""" (description)

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

import numpy as np
def align_common_frame(struc: np.ndarray) -> np.ndarray:
    """Aligns a structure in a common reference frame    
    Input: 
      struc: backbone structure of shape MxBx3
      B is the number of backbone atoms, 
        This must be 4, and in the order N, CA, C, O
      M is the number of residues
    Output:
      backbone structure of shape MxBx3

    In the reference frame:
    - The CA atom is at the origin
    - The Y-axis is the C-N vector
    - The Z-axis is perpendicular to the N-CA-C plane
    """
    assert struc.ndim == 3
    assert struc.shape[1] == 4
    assert struc.shape[2] == 3
    result = struc.copy()    
    pp = result[0]
    result -= pp[1]
    r = np.zeros((3,3))
    vec1 = pp[0] - pp[1]  # CA-N vector
    vec2 = pp[2] - pp[1]  # CA-C vector
    vec3 = np.cross(vec1, vec2) # Z axis
    vec1 /= np.sqrt((vec1*vec1).sum())
    vec2 /= np.sqrt((vec2*vec2).sum())
    vec3 /= np.sqrt((vec3*vec3).sum())
    vec4 = vec1 - vec2     # C-N vector (Y axis)
    vec4 /= np.sqrt((vec4*vec4).sum())
    r[:,2] = vec3
    r[:,1] = vec4
    r[:,0] = np.cross(r[:,1], r[:,2])
    r[:,0] /= np.sqrt((r[:,0]*r[:,0]).sum())
    result2 = result.reshape(-1, 3) # view into array "result"
    result2[:] = np.dot(result2, r)  # This modifies "result" as well
    return result