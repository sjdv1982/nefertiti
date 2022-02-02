if __name__ == "__main__":
    import numpy as np
    from nefertiti.protocols.greedy import greedy_backbone_rmsd

data = []
for n in range(100):
    d = {
        "name": "octa{}".format(n+1),        
        "file": "../benchmarks/octacommon-aligned.npy",
        "index": n,
        "format": "npy",
    }
    data.append(d)

for n in range(100):
    d = {
        "name": "dodeca{}".format(n+1),
        "file": "../benchmarks/dodecacommon-aligned.npy",
        "index": n,
        "format": "npy",
    }
    data.append(d)


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

if __name__ == "__main__":
    fraglib = np.load("../fraglib/dummy.npy")
    import itertools
    alldata = itertools.product(data, poolsizes)
    import multiprocessing
    pool = multiprocessing.Pool(8) # some parallelization is already inside the Nefertiti code
    pool.starmap(run, alldata)  