"""Progressions related to coordinate calculation

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""
from ..functions.matrix import dotmat


from ..MainState import Stage
def update_coordinates_backbone(
    s: Stage, newsize
) -> None:
    size = s.size
    fraglib = s.fraglib
    frags = fraglib.coor.backbone4_centered
    fraglen, bbsize = frags.shape[1:3]
    frags = frags.reshape(len(frags), -1, 4)
    coor = s.coor.backbone
    assert coor is not None  #.backbone
    mats = s.matrices[size:newsize]
    traj = s.trajectories[size:newsize, -1]
    newcoor = dotmat(mats, frags, traj)[:, :, :3]
    newcoor = newcoor.reshape(len(newcoor), fraglen, bbsize, 3)
    coor[size:newsize] = newcoor