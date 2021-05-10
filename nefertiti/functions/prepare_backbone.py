""" (description)

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

import numpy as np
def prepare_backbone(struc, fraglen, bblen=4):
    """Prepares a backbone structure for RMSD calculation
    Input: 
      struc: backbone structure of shape MxBx3 or B*Mx3, where B is bblen
      M is the number of residues
      fraglen: length (in residues) of the backbone fragment library
      bblen: the number of backbone atoms (by default 4, i.e. N, CA, C, O)
    Output:
    - Structure of shape MxBx4, in (x,y,z,1) form
    - Structure of shape KxFxBx4, where F is fraglen and B is bblen
      The structure is in (x,y,z,1) form
      K = M-F+1
      each fragment k in K contains the coordinates for residue k:k+F
      
      This means that residues beyond the first (and before the last) get duplicated
      The structure is centered so that the average atom (after duplication!) is (0,0,0)
    
    - Residuals (sum of squares) of the centered structure
      K residuals are returned. Residual n is for the structure up to fragment n
    """
    refe = struc.reshape(-1, bblen, 3)
    k = len(refe) - fraglen + 1
    result = np.zeros((k, fraglen, bblen, 3))
    for kk in range(k):
        result[kk] = refe[kk:kk+fraglen]
    com = result.reshape(-1, 3).mean(axis=0)
    result -= com
    residuals = (result.reshape(k,-1)**2).sum(axis=1)
    residuals = np.cumsum(residuals)
    refe4 = np.ones(refe.shape[:-1]+(4,))
    refe4[:, :, :3] = refe
    refe_frag4 =  np.ones(result.shape[:-1]+(4,))
    refe_frag4[:, :, :, :3] = result
    return refe4, refe_frag4, residuals