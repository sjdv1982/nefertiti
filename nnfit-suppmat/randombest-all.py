if __name__ == "__main__":
    import numpy as np
    from nefertiti.protocols.randombest import randombest_backbone_rmsd

ntraj = 1000
data = []
for n in range(100):
    d = {
        "name": "octa{}".format(n+1),        
        "file": "../benchmarks/octacommon-aligned.npy",
        "index": n,
        "format": "npy",
        "best_of_factor": 1000,
        "redundancy": 200,
    }
    data.append(d)

for n in range(100):
    d = {
        "name": "dodeca{}".format(n+1),
        "file": "../benchmarks/dodecacommon-aligned.npy",
        "index": n,
        "format": "npy",
        "best_of_factor": 1000,
        "redundancy": 200,
    }
    data.append(d)


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

if __name__ == "__main__":
    fraglib = np.load("../fraglib/dummy.npy")
    import multiprocessing
    pool = multiprocessing.Pool(8) # some parallelization is already inside the Nefertiti code
    pool.map(run, data)  