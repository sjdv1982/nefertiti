"""Progressions to enumerate the k-best fragment trajectories 

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

from ..MainState import MainState
from .grow import grow, sort_score, filter_score, filter_score_unsorted
import numpy as np

import time
import logging
logger = logging.getLogger("nefertiti")

from typing import List

def kbest(
    s: MainState,
    scorer: callable,
    updaters:list,
    k,
    *,
    minblocksize:int,
    upper_estimate:float,
    downstream_best:List[float],
) -> None:

    #s.stages[-1].score_threshold = upper_estimate

    s1 = s.stages[0]
    grow(None, s1, scorer=scorer, updaters=updaters)
    sort_score(s1)

    nstages = len(s.stages)
    if nstages == 1:
        return

    cutoff = None
    delta = 0.1
    def propagate_threshold(stagenr, threshold, final_threshold):
        nonlocal cutoff
        if stagenr == nstages - 1:
            if cutoff is not None:
                assert threshold <= cutoff
                d = cutoff - threshold
                old_cutoff = cutoff
                cutoff = threshold
                if d < delta:
                    return
                print("CUTOFF", cutoff, old_cutoff)
            else:
                cutoff = threshold
                print("CUTOFF", cutoff)
        else:
            stage = s.stages[stagenr]
            curr_threshold = stage.score_threshold
            if curr_threshold is not None:
                if threshold >= curr_threshold:
                    return
                d = curr_threshold - threshold
                stage.score_threshold = threshold       
                if d < delta:
                    return
            else:
                stage.score_threshold = threshold
            print("THRESHOLD", stagenr+1, threshold)
            filter_score(stage)
        if stagenr > 0:
            next_threshold = final_threshold - downstream_best[stagenr-1]
            propagate_threshold(stagenr-1, next_threshold, final_threshold)        
    
    while 1:
        progress = False
        for curr_minblocksize in (minblocksize, 1):
            if curr_minblocksize == 1 and progress:
                break
            for n in range(nstages-2, -1, -1):
                if s.stages[n].size >= curr_minblocksize:
                    break
            else:
                continue

            progress = True
            s1, s2 = s.stages[n], s.stages[n+1]
            if n+1 == nstages-1:
                print("FIN", nstages, s1.scores[0], s1.score_threshold)
            else:
                print("GROW", nstages, n+1, s1.size, n+2, s2.size, s1.scores[0], s1.score_threshold)
            old_size = s2.size
            grow(s1, s2, scorer=scorer, updaters=updaters)
            if n+1 == nstages-1:
                filter_score_unsorted(s2, old_size)
                if s2.size > old_size and s2.size >= k:
                    sort_score(s2)              
                    threshold = s2.scores[k-1]
                    propagate_threshold(nstages-1, threshold, threshold)
                    s2.size = k
            else:
                sort_score(s2)
                filter_score(s2)            
                print("POST", nstages, n+1, s1.size, n+2, s2.size)
        if not progress:
            break