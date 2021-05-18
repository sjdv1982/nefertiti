"""Progressions to randomly generate fragment trajectories
within an (optional) score threshold

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

from ..MainState import MainState
from .grow import grow_random, sort_score, filter_score_unsorted
import numpy as np
import time
import logging
logger = logging.getLogger("nefertiti")

from typing import List

def randombest(
    s: MainState,
    scorer: callable, #or None
    updaters:list,
    *,
    ntrajectories=int,
    threshold:float, # or None
    downstream_best:List[float], # or None
    minblocksize:int,
) -> None:

    nstages = len(s.stages)
    assert s.stages[-1].maxsize >= ntrajectories

    if downstream_best is not None:
        assert threshold is not None
        assert len(downstream_best) == nstages - 1
        t = threshold
        for n in range(nstages-1, -1, -1):
            stage = s.stages[n]
            stage.score_threshold = t
            t -= downstream_best[n-1]
    else:
        s.stages[-1].score_threshold = threshold
        
    def report():
        print(curr_time - start)
        for n in range(nstages):
            stage = s.stages[n]
            score_threshold = 999
            if stage.score_threshold is not None:
                score_threshold = stage.score_threshold
            print("{:d} {:6d} {:.3f}".format(n+1, stage.size, score_threshold))
        print()

    start = time.time()
    last_time = start
    first_stage = s.stages[0]
    last_stage = s.stages[-1]
    while last_stage.size < ntrajectories:

        curr_time = time.time()
        if curr_time - last_time > 5:
            report()
            last_time = curr_time
        
        if first_stage.size < minblocksize:
            old_size = first_stage.size
            progress = True
            grow_random(None, first_stage, scorer=scorer, updaters=updaters)
            filter_score_unsorted(first_stage, old_size)
            continue

        progress = False
        for curr_minblocksize in (minblocksize, 1):
            if curr_minblocksize == 1 and progress:
                break
            for n in range(nstages-2, -1, -1):
                s1, s2 = s.stages[n], s.stages[n+1]
                if s1.size < curr_minblocksize:
                    continue
                space = s2.maxsize - s2.size
                if space // s.fraglib.nfrags < curr_minblocksize:
                    continue
                break
            else:
                continue

            progress = True
            old_size = s2.size
            s1, s2 = s.stages[n], s.stages[n+1]
            grow_random(s1, s2, scorer=scorer, updaters=updaters)
            filter_score_unsorted(s2, old_size)
    sort_score(s.stages[-1])