"""Progressions to builds fragment trajectories in a greedy way

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

from ..MainState import MainState
from .grow import grow, sort_score
import time
import logging
logger = logging.getLogger("nefertiti")

def greedy(
    s: MainState,
    scorer: callable,
    updaters:list,
    poolsize,
) -> None:
    s1 = s.stages[0]
    grow(None, s1, scorer=scorer, updaters=updaters)
    sort_score(s1)
    if s1.size > poolsize:
        s1.size = poolsize

    active = 0
    it = 0
    t = time.time()
    while active < len(s.stages) - 1:
        it += 1

        s1 = s.stages[active]
        s2 = s.stages[active+1]
        if s1.size == 0:
            logger.info(
                "Greedy done: %d/%d, %.3f sec", 
                active+1, time.time()-t,
                
            )
            active += 1
            continue

        #print("GREEDY GROW", active+1)
        grow(s1, s2, scorer=scorer, updaters=updaters)            
        sort_score(s2)
        if s2.size > poolsize:
            s2.size = poolsize

        """
        # Works only for backbone RMSD growing...    
        if s2.size:
            natoms = (active + 2) * s.fraglib.fraglen * len(s.bb_atoms) 
            import numpy as np 
            best_rmsd = np.sqrt(s2.scores[0] / natoms)
            print(active+2, best_rmsd)
            print()
        # /
        """
