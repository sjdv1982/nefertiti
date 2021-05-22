from seamless.highlevel import Context, Cell, Module
from seamless.stdlib import stdlib

import numpy as np
fraglib = np.load("../fraglib/dummy.npy")
refe = np.load("../benchmarks/octacommon-aligned.npy")[0]

ctx = Context()
ctx.include(stdlib.python_package)

ctx.package_dirdict = Cell("plain")
ctx.package_dirdict.mount("../nefertiti", as_directory=True, mode="r")
ctx.package_dirdict_value = ctx.package_dirdict
ctx.package_dirdict_value.celltype = "plain"
ctx.package_dirdict_value.mount("package_dirdict.json", mode="w")

ctx.package = Cell("plain")
ctx.package.mount("package.json", mode="w")

ctx.python_package = ctx.lib.python_package(
    package_dirdict = ctx.package_dirdict,
    package_name = "nefertiti",
    package = ctx.package
)

ctx.module = Module()
ctx.module.code = ctx.package
ctx.compute()
graph = ctx.get_graph()
zip = ctx.get_zip()

def kbest_test(refe, fraglib, k):
    import numpy as np
    from .nefertiti.protocols.kbest import kbest_backbone_rmsd

    import logging
    logging.basicConfig()
    logging.getLogger("nefertiti").setLevel(logging.INFO)

    traj, rmsd = kbest_backbone_rmsd(
        refe, fraglib,
        format="npy",
        k=k
    )
    print(rmsd[:30], rmsd[-10:])
    return {
        "traj": traj,
        "rmsd": rmsd,
    }

ctx.kbest_test = kbest_test
ctx.kbest_test.nefertiti = ctx.module
ctx.kbest_test.refe = refe
ctx.kbest_test.pins.refe.celltype = "binary"
ctx.kbest_test.fraglib = fraglib
ctx.kbest_test.pins.fraglib.celltype = "binary"
ctx.kbest_test.k = 1000
ctx.kbest_test.pins.k.celltype = "int"
ctx.compute()

if ctx.kbest_test.status != "Status: OK":
    print(ctx.kbest_test.status)
    exc = ctx.kbest_test.exception
    if isinstance(exc, dict) and list(exc.keys()) == ["nefertiti"]:
        exc = exc["nefertiti"]
    print(exc)
    print(ctx.kbest_test.logs)

    import sys
    sys.exit()

import os, json
json.dump(graph, open("nefertiti.seamless", "w"), sort_keys=True, indent=2)
with open("nefertiti.zip", "bw") as f:
    f.write(zip)
print(ctx.kbest_test.logs)
print("Graph saved")