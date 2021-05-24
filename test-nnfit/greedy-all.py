import numpy as np
from nefertiti.protocols.greedy import greedy_backbone_rmsd

data = [
    {
        "name": "octa1",        
        "file": "../benchmarks/octacommon-aligned.npy",
        "index": 1,
        "format": "npy",
    },
    {
        "name": "octa4",        
        "file": "../benchmarks/octacommon-aligned.npy",
        "index": 3,
        "format": "npy",
    },
    {
        "name": "octa17",        
        "file": "../benchmarks/octacommon-aligned.npy",
        "index": 16,
        "format": "npy",
    },

    {
        "name": "dodeca1",        
        "file": "../benchmarks/dodecacommon-aligned.npy",
        "index": 0,
        "format": "npy",
    },
    {
        "name": "dodeca14",        
        "file": "../benchmarks/dodecacommon-aligned.npy",
        "index": 13,
        "format": "npy",
    },

    {
        "name": "casp14",        
        "file": "casp14-T1033-D1.pdb",
        "format": "pdb",
    },

    {
        "name": "trypsin",        
        "file": "1AVXA-unbound-heavy.pdb",
        "format": "pdb",
    },
]

fraglib = np.load("../fraglib/dummy.npy")

poolsizes = 100, 200, 500, 1000
def run(d, poolsize):
    f = d["file"]
    if d["format"] == "pdb":
        refe = open(f).read()
    else:
        refe = np.load(f)
        if d.get("index") is not None:
            refe = refe[d["index"]]
    print(d["name"], poolsize)
    trajectories, rmsds = greedy_backbone_rmsd(
        refe, fraglib,
        format=d["format"],
        poolsize=poolsize
    )
    print("/" + d["name"], poolsize)
    assert len(trajectories) == len(rmsds) == poolsize
    outpattern = "data/greedy-" + d["name"] + "-poolsize-%d" % poolsize
    np.save(outpattern + "-traj.npy", trajectories)
    np.save(outpattern + "-rmsd.npy", rmsds)

import itertools
alldata = itertools.product(data, poolsizes)
import multiprocessing
pool = multiprocessing.Pool()
pool.starmap(run, alldata)  