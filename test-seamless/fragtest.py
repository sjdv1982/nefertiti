import numpy as np
import json
import seamless
from seamless.highlevel import Context
from seamless.highlevel.Module import Module

ctx = Context()

# Import nefertiti (could use some syntactic sugar)
nctx = Context()
nctx.add_zip("../seamless-dist/nefertiti.zip")
nctx.set_graph(
    json.load(open("../seamless-dist/nefertiti.seamless")),
    #mounts=False
)

ctx.ctx_nefertiti = nctx
ctx.compute() 
ctx.nefertiti = Module()
ctx.nefertiti.code = ctx.ctx_nefertiti.module.code
ctx.compute()
# /import nefertiti 

fraglib0 = np.load("../fraglib/dummy.npy")
print(fraglib0.shape)


traj0="""33  56  24  24  15  10  51  33  46  46  45  27  78  15   1   1   1  67
  47  81  26 103  39   2  13  13  82 105  65  89  31  16   1  51  10  13
  52   0  33   7 100  54  30   1   1   1   1  32  33   7  87  65  20  27
  28  45  87  83  27 100   9   1  15   1   1  87  78   9   1   1  82  77
  27  15   1 100   0  43  12  66  33  20   1  82  82   1   1   9  18  50
  24  24  20  20  67  81  27  19   1   1   1   1  10  20  45  50  24   7
  15  10  27  28   1  15   1   1  82   1  87  67  10  20  88   2  11  93
  26   1   1   1  67  30  10  20   1   1  82  15   1   1   1   9  55  37
   3  23   3   3  27  50  45  33 108   8  52  54  27  78   1  10  87   1
  87  25 106   7 110  97  79   6 107  64   5  57  58  33  15  10   2  32
  13  99  42  28  24  14  78   1  10  85 100   1  89  16  63  59  60  91
  31  16   1  10  82   1  82   1 102   3  23   3  58   3  27   3  23   3
  23   3  27  27"""
traj = np.array([int(v) for v in traj0.split()])

def func(fraglib0, traj):
    pass
    return 42

ctx.tf = func
ctx.tf.code.mount("tf-fragtest.py", "r")
ctx.tf.traj = traj
ctx.tf.pins.traj.celltype = "binary"
ctx.tf.fraglib0 = fraglib0
ctx.tf.pins.fraglib0.celltype = "binary"
ctx.tf.nefertiti = ctx.nefertiti
ctx.compute()

print(ctx.tf.status)
print(ctx.tf.exception)
pdbdata = ctx.tf.result.value.unsilk

with open("fragtest.pdb", "w") as f:
    f.write(pdbdata)
