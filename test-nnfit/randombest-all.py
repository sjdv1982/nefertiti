import numpy as np
from nefertiti.protocols.randombest import randombest_backbone_rmsd

ntraj = 1000
data = [
    {
        "name": "octa1",        
        "file": "../benchmarks/octacommon-aligned.npy",
        "index": 1,
        "format": "npy",
        "best_of_factor": 1000,
        "redundancy": 200,
    },
    {
        "name": "octa4",        
        "file": "../benchmarks/octacommon-aligned.npy",
        "index": 3,
        "format": "npy",
        "best_of_factor": 1000,
        "redundancy": 200,
    },
    {
        "name": "octa17",        
        "file": "../benchmarks/octacommon-aligned.npy",
        "index": 16,
        "format": "npy",
        "best_of_factor": 1000,
        "redundancy": 200,
    },

    {
        "name": "dodeca1",        
        "file": "../benchmarks/dodecacommon-aligned.npy",
        "index": 0,
        "format": "npy",
        "best_of_factor": 1000,
        "redundancy": 200,
    },
    {
        "name": "dodeca14",        
        "file": "../benchmarks/dodecacommon-aligned.npy",
        "index": 13,
        "format": "npy",
        "best_of_factor": 1000,
        "redundancy": 200,
    },

    {
        "name": "casp14",        
        "file": "casp14-T1033-D1.pdb",
        "format": "pdb",
        "best_of_factor": 100,
        "redundancy": 200,
    },

    {
        "name": "trypsin",        
        "file": "1AVXA-unbound-heavy.pdb",
        "format": "pdb",
        "best_of_factor": 100,
        "redundancy": 200,
    },
]

fraglib = np.load("../fraglib/dummy.npy")

def run(d):
    f = d["file"]
    if d["format"] == "pdb":
        refe = open(f).read()
    else:
        refe = np.load(f)
        if d.get("index") is not None:
            refe = refe[d["index"]]
    print(d["name"])
    best_of_factor = d["best_of_factor"]
    redundancy = d["redundancy"]
    trajectories, rmsds = randombest_backbone_rmsd(
        refe, fraglib,
        format=d["format"],
        ntrajectories=redundancy*best_of_factor,
        use_downstream_best=False,
        max_rmsd=None
    )
    assert len(trajectories) == len(rmsds) == redundancy*best_of_factor
    del trajectories
    
    threshold = rmsds[redundancy-1]
    print(d["name"], threshold)

    ntraj=1000
    trajectories, rmsds = randombest_backbone_rmsd(
        refe, fraglib,
        format=d["format"],
        ntrajectories=ntraj,
        max_rmsd=threshold,
        use_downstream_best=False
    )
    assert len(trajectories) == len(rmsds) == ntraj

    outpattern = "data/randombest-" + d["name"]
    open(outpattern+"-threshold.txt", "w").write("%.6f" % threshold)
    np.save(outpattern + "-traj.npy", trajectories)
    np.save(outpattern + "-rmsd.npy", rmsds)
    print("/" + d["name"])

import multiprocessing
pool = multiprocessing.Pool()
pool.map(run, data)  