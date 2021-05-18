"""Progressions to enumerate the k-best fragment trajectories 

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

from ..MainState import MainState
from .grow import grow, sort_score, filter_score_sorted, filter_score_unsorted
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

    def report():
        print(curr_time - start)
        for n in range(nstages):
            stage = s.stages[n]
            score_threshold = 999
            if n == nstages-1 and cutoff is not None:
                score_threshold = cutoff
            elif stage.score_threshold is not None:
                score_threshold = stage.score_threshold
            print("{:d} {:6d} {:.3f}".format(n+1, stage.size, score_threshold))
        print()

    cutoff = None
    start = time.time()
    s1 = s.stages[0]
    grow(None, s1, scorer=scorer, updaters=updaters)
    sort_score(s1)

    nstages = len(s.stages)
    if nstages == 1:
        return

    has_new_cutoff = False
    delta = 0.1
    def propagate_threshold(stagenr, threshold, final_threshold):
        nonlocal cutoff, has_new_cutoff
        if stagenr == nstages - 1:
            if cutoff is not None:
                assert threshold <= cutoff
                d = cutoff - threshold
                old_cutoff = cutoff
                cutoff = threshold
                if d < delta:
                    return
                has_new_cutoff = True
            else:
                cutoff = threshold
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
            filter_score_sorted(stage)
        if stagenr > 0:
            next_threshold = final_threshold - downstream_best[stagenr-1]
            propagate_threshold(stagenr-1, next_threshold, final_threshold)        

    if upper_estimate is not None:
        propagate_threshold(nstages-1, upper_estimate, upper_estimate)

    last_time = start
    while 1:
        curr_time = time.time()
        if curr_time - last_time > 5:
            report()
            last_time = curr_time
        progress = False
        optblocksize = 100
        optblocksize2 = 5
        for curr_minblocksize in (minblocksize, 1):
            if curr_minblocksize == 1 and progress:
                break
            for n in range(nstages-2, -1, -1):
                if n < nstages-2:
                    s1, s2 = s.stages[n], s.stages[n+1]
                    space = s2.maxsize - s2.size
                    if space // s.fraglib.nfrags < curr_minblocksize:
                        continue
                if s.stages[n].size >= optblocksize:
                    break
                if k == 1 and not has_new_cutoff:
                    if s.stages[n].size >= optblocksize2:
                        break                    
            else:
                for n in range(0, nstages-1):
                    if n < nstages-2:
                        s1, s2 = s.stages[n], s.stages[n+1]
                        space = s2.maxsize - s2.size
                        if space // s.fraglib.nfrags < curr_minblocksize:
                            continue
                    if s.stages[n].size >= curr_minblocksize:
                        break
                else:
                    continue

            progress = True
            s1, s2 = s.stages[n], s.stages[n+1]
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
                filter_score_sorted(s2)            
        if not progress:
            break
    s.downstream_best_score = [s.stages[-1].scores[0]] + downstream_best