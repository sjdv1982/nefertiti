"""Progressions to builds fragment trajectories in a greedy way

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

from ..MainState import MainState
from .grow import grow

def greedy(
    s: MainState,
    scorer: callable,
    updaters:list,
) -> None:
    grow(None, s.stages[0], scorer=scorer, updaters=updaters)