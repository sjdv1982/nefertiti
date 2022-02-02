if __name__ == "__main__":
    import numpy as np
    from nefertiti.protocols.kbest import kbest_backbone_rmsd

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
    del rmsds, trajectories
    import gc
    gc.collect()
    return

if __name__ == "__main__":
    fraglib = np.load("../fraglib/dummy.npy")
    import multiprocessing
    pool = multiprocessing.Pool(16) # some parallelization is already inside the Nefertiti code
    pool.map(run, data)  