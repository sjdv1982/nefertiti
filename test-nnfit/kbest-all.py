import numpy as np
from nefertiti.protocols.kbest import kbest_backbone_rmsd

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
]

fraglib = np.load("../fraglib/dummy.npy")

k = 100000  # for smaller k, just take the top
def run(d):
    f = d["file"]
    if d["format"] == "pdb":
        refe = open(f).read()
    else:
        refe = np.load(f)
        if d.get("index") is not None:
            refe = refe[d["index"]]
    print(d["name"])
    trajectories, rmsds = kbest_backbone_rmsd(
        refe, fraglib,
        format=d["format"],
        k=k
    )

    print("/" + d["name"])
    assert len(trajectories) == len(rmsds) == k
    outpattern = "data/kbest-" + d["name"] + "-k-%d" % k
    np.save(outpattern + "-traj.npy", trajectories)
    np.save(outpattern + "-rmsd.npy", rmsds)

import multiprocessing
pool = multiprocessing.Pool()
pool.map(run, data)  