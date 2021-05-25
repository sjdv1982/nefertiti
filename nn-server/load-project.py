
PROJNAME = "nn-server"

import os, sys, shutil
import json
import numpy as np

from seamless.highlevel import Context, Cell, Transformer, Module

import seamless
seamless.database_sink.connect()
seamless.database_cache.connect()

ctx = None
webctx = None
save = None

def pr(*args):
    print(*args, file=sys.stderr)

async def define_graph(ctx):
    """Code to define the graph
    Leave this function empty if you want load() to load the graph from graph/PROJNAME.seamless 
    """
    # Import nefertiti (could use some syntactic sugar)
    nctx = Context()
    nctx.add_zip("../seamless-dist/nefertiti.zip")
    nctx.set_graph(
        json.load(open("../seamless-dist/nefertiti.seamless")),
        mounts=False
    )

    ctx.ctx_nefertiti = nctx
    await ctx.computation() 
    ctx.nefertiti = Module()
    ctx.nefertiti.code = ctx.ctx_nefertiti.module.code
    await ctx.computation()
    # /import nefertiti 


    #ctx.params0 = Cell("plain").mount("mounts/params0.json")
    ctx.params0 = Cell() 
    ctx.pdb = Cell("text").mount("mounts/input.pdb")
    ctx.pdb.share(readonly=False)

    def load_pdb(pdbdata):        
        from .nefertiti.functions.parse_pdb import parse_pdb, get_backbone, get_xyz, get_sequence
        struc = parse_pdb(pdbdata)
        bb = get_backbone(struc, ["N", "CA", "C", "O"])
        bb_coor = get_xyz(bb)
        return bb_coor, len(bb_coor)
    ctx.load_pdb = load_pdb
    ctx.load_pdb.pdbdata = ctx.pdb
    ctx.load_pdb.nefertiti = ctx.nefertiti
    ctx.load_pdb_result = ctx.load_pdb
    
    ctx.params0a = Cell()
    ctx.params0a = ctx.params0
    ctx.params0b = Cell()
    ctx.params0b["high-rmsd"] = ctx.params0a["high-rmsd"]
    ctx.params0b["low-rmsd"] = ctx.params0a["low-rmsd"]
    ctx.params0b["nresidues"] = ctx.load_pdb_result[1]
    ctx.params = Cell("plain").mount("mounts/params.json", "w")
    ctx.params = ctx.params0b

    def validate_cost(params):
        from seamless.core.transformation import SeamlessTransformationError
        from .nefertiti.utils.nn_cost import nn_cost
        has_err, reason = nn_cost(params)
        if has_err:
            raise SeamlessTransformationError(reason)
        return params
        
    ctx.validate_cost = validate_cost
    ctx.validate_cost.params = ctx.params
    ctx.validate_cost.nefertiti = ctx.nefertiti
    ctx.validated_params = ctx.validate_cost

    ctx.fraglib = Cell("binary").set(np.load("../fraglib/dummy.npy"))
    ctx.nfraglib = 111
    ctx.fraglen = 4

    def do_run(params):
        run_threshold = False
        run_random = False
        run_greedy = False
        run_nn = False

        phigh2 = params["high-rmsd"]["analysis"]
        high_mode = phigh2["mode"]
        high_mode = high_mode.split(",")
        if len(high_mode) == 1:
            high_fit_mode = high_show_mode = high_mode[0]
        else:
            high_fit_mode, high_show_mode = high_mode
        if high_fit_mode in ("threshold", "random") or high_show_mode in ("threshold", "random"):
            run_threshold = True
            if high_fit_mode == "random" or high_show_mode == "random":
                run_random = True    
        

        plow2 = params["low-rmsd"]["analysis"]
        low_mode = plow2["mode"]
        low_mode = low_mode.split(",")
        if len(low_mode) == 1:
            low_fit_mode = low_show_mode = low_mode[0]
        else:
            low_fit_mode, low_show_mode = low_mode
        if low_fit_mode == "greedy" or low_show_mode == "greedy":
            run_greedy = True
        if low_fit_mode == "nn" or low_show_mode == "nn":
            run_nn = True

        return {
            "run_greedy": run_greedy,
            "run_nn": run_nn,
            "run_threshold": run_threshold,
            "run_random": run_random,
        }
    ctx.do_run = do_run
    ctx.do_run.params = ctx.validated_params
    ctx.do_run_result = ctx.do_run
        

    def calc_greedy(do_run, refe, fraglib, poolsize):
        if not do_run:
            return -1
        from .nefertiti.protocols.greedy import greedy_backbone_rmsd
        _, rmsds = greedy_backbone_rmsd(
            refe.unsilk, fraglib.unsilk,
            format="npy",
            poolsize=poolsize
        )
        return rmsds
    ctx.calc_greedy = calc_greedy
    ctx.calc_greedy.nefertiti = ctx.nefertiti
    ctx.calc_greedy.do_run = ctx.do_run_result["run_greedy"]
    ctx.calc_greedy.refe = ctx.load_pdb_result[0]
    ctx.calc_greedy.fraglib = ctx.fraglib
    ctx.calc_greedy.poolsize = ctx.validated_params["low-rmsd"]["computation"]["greedy"]["poolsize"]


    def calc_nn(do_run, refe, fraglib, k):
        from .nefertiti.protocols.kbest import kbest_backbone_rmsd
        if not do_run:
            return -1
        _, rmsds = kbest_backbone_rmsd(
            refe.unsilk, fraglib.unsilk,
            format="npy",
            k=k
        )  
        return rmsds  
        
    ctx.calc_nn = calc_nn
    ctx.calc_nn.nefertiti = ctx.nefertiti
    ctx.calc_nn.do_run = ctx.do_run_result["run_nn"]
    ctx.calc_nn.refe = ctx.load_pdb_result[0]
    ctx.calc_nn.fraglib = ctx.fraglib
    ctx.calc_nn.k = ctx.validated_params["low-rmsd"]["computation"]["near-native"]["k"]
    ctx.calc_nn_result = ctx.calc_nn

    def calc_threshold(do_run, refe, fraglib, best_of_factor, redundancy):
        if not do_run:
            return -1
        from .nefertiti.protocols.randombest import randombest_backbone_rmsd
        _, rmsds = randombest_backbone_rmsd(
            refe.unsilk, fraglib.unsilk,
            format="npy",
            ntrajectories=best_of_factor * redundancy,
            use_downstream_best=False,
            max_rmsd=None
        )
        threshold = rmsds[redundancy-1] 
        
        return threshold
    ctx.calc_threshold = calc_threshold
    ctx.calc_threshold.nefertiti = ctx.nefertiti
    ctx.calc_threshold.do_run = ctx.do_run_result["run_threshold"]
    ctx.calc_threshold.refe = ctx.load_pdb_result[0]
    ctx.calc_threshold.fraglib = ctx.fraglib
    ctx.calc_threshold.best_of_factor = ctx.validated_params["high-rmsd"]["computation"]["threshold"]["best_of_factor"]
    ctx.calc_threshold.redundancy = ctx.validated_params["high-rmsd"]["computation"]["threshold"]["redundancy"]
    ctx.threshold = ctx.calc_threshold


    def calc_random(do_run, refe, fraglib, nstruc, threshold):
        if not do_run:
            return -1
        from .nefertiti.protocols.randombest import randombest_backbone_rmsd
        _, rmsds = randombest_backbone_rmsd(
            refe.unsilk, fraglib.unsilk,
            format="npy",
            ntrajectories=nstruc,
            use_downstream_best=False,
            max_rmsd=threshold
        )
        return rmsds

    ctx.calc_random = calc_random
    ctx.calc_random.nefertiti = ctx.nefertiti
    ctx.calc_random.do_run = ctx.do_run_result["run_threshold"]
    ctx.calc_random.refe = ctx.load_pdb_result[0]
    ctx.calc_random.fraglib = ctx.fraglib
    ctx.calc_random.nstruc = ctx.validated_params["high-rmsd"]["computation"]["random"]["nstructures"]
    ctx.calc_random.threshold = ctx.threshold


    ctx.data = Cell()
    await ctx.translation()
    ctx.data.set({}) 
    ctx.calc_greedy_result = ctx.calc_greedy
    ctx.data.greedy = ctx.calc_greedy_result
    ctx.data.threshold = ctx.threshold
    ctx.calc_random_result = ctx.calc_random
    ctx.data.random = ctx.calc_random_result
    ctx.data["near-native"] = ctx.calc_nn_result

    await ctx.computation()

    def fit_nn(data, params, nfraglib, fraglen):
        import numpy as np
        from .nefertiti.utils.fit_nn import fit_nn #as _fit_nn (BUG in Seamless)
        from silk import Silk
        if isinstance(data, Silk):
            data = data.unsilk
        if isinstance(params, Silk):
            params = params.unsilk
        result = fit_nn(params, data, nfraglib, fraglen)
        result["plot"] = np.array(result["plot"]) #BUG in seamless
        return result

    ctx.fit_nn = fit_nn
    ctx.fit_nn.nefertiti = ctx.nefertiti
    ctx.fit_nn.nfraglib = ctx.nfraglib
    ctx.fit_nn.fraglen = ctx.fraglen
    ctx.fit_nn.data = ctx.data
    ctx.fit_nn.params = ctx.params

    ctx.result = ctx.fit_nn
    ctx.plot0 = ctx.result["plot"]
    ctx.plot = ctx.plot0
    ctx.plot.celltype = "bytes"
    ctx.plot.share()
    ctx.plot.mimetype = "png"
    ctx.plot.mount("mounts/plot.png", "w")   


    ctx.rmsd1 = Cell("float").set(1)
    ctx.rmsd2 = Cell("float").set(2)
    def calc_specificity(equation, rmsd1, rmsd2):
        from math import sqrt, log, exp
        result = equation["text"] + "\n"
        a, c = equation["a"], equation["c"]
        p, q = equation["p"], equation["q"]
        logN2_1 = p * rmsd1 + q
        logN2_2 = p * rmsd2 + q
        if logN2_1 < 0:
            result += "RMSD 1 is below the lowest RMSD"
            return result
        if logN2_2 < 0:
            result += "RMSD 2 is below the lowest RMSD"
            return result
        logN_1 = sqrt(logN2_1)
        logN_2 = sqrt(logN2_2)
        if logN_1 > logN_2:
            word = "less"
            logNa, logNb = logN_2, logN_1 
        else:
            word = "more"
            logNa, logNb = logN_1, logN_2
        d = (logNb - logNa) / log(10)
        if d < 4:
            txt = "%d times" % (int(10**d)+0.5)
        else:
            txt = "%.2f orders of magnitude" % d
        result += "RMSD 1 is %s %s specific than RMSD 2" % (txt, word)
        return result
    ctx.calc_specificity = calc_specificity
    ctx.calc_specificity.rmsd1 = ctx.rmsd1 
    ctx.calc_specificity.rmsd2 = ctx.rmsd2
    ctx.calc_specificity.equation = ctx.result["equation"]

    ctx.rmsd1.share(readonly=False)
    ctx.rmsd2.share(readonly=False)
    ctx.specificity = ctx.calc_specificity
    ctx.specificity.celltype = "text"
    ctx.specificity.share()    
    
    await ctx.translation()

    c = ctx.high_comp_random_nstructures = Cell("int").set(1000)
    c.share(readonly=False)
    ctx.params0["high-rmsd"].computation.random.nstructures = c

    c = ctx.high_comp_threshold_factor = Cell("int").set(1000)
    c.share(readonly=False)
    ctx.params0["high-rmsd"].computation.threshold.best_of_factor = c

    c = ctx.high_comp_threshold_redundancy = Cell("int").set(200)
    c.share(readonly=False)
    ctx.params0["high-rmsd"].computation.threshold.redundancy = c

    c = ctx.high_ana_mode = Cell("str").set("random")
    c.share(readonly=False)
    ctx.params0["high-rmsd"].analysis.mode = c

    c = ctx.high_ana_random_bins = Cell("int").set(10)
    c.share(readonly=False)
    ctx.params0["high-rmsd"].analysis.random.bins = c

    c = ctx.high_ana_random_mode = Cell("str").set("binning")
    c.share(readonly=False)
    ctx.params0["high-rmsd"].analysis.random.mode = c

    c = ctx.high_ana_random_discard_lower = Cell("int").set(50)    
    c.share(readonly=False)
    ctx.params0["high-rmsd"].analysis.random.discard_lower = c

    c = ctx.high_ana_random_discard_upper = Cell("int").set(0)
    c.share(readonly=False)
    ctx.params0["high-rmsd"].analysis.random.discard_upper = c
    
    c = ctx.low_comp_greedy_poolsize = Cell("int").set(500)
    c.share(readonly=False)
    ctx.params0["low-rmsd"].computation.greedy.poolsize = c

    c = ctx.low_comp_nn_k = Cell("int").set(100000)
    c.share(readonly=False)
    ctx.params0["low-rmsd"].computation["near-native"].k = c

    c = ctx.low_ana_mode = Cell("str").set("nn,greedy")
    c.share(readonly=False)
    ctx.params0["low-rmsd"].analysis.mode = c

    c = ctx.low_ana_greedy_bins = Cell("int").set(10)
    c.share(readonly=False)
    ctx.params0["low-rmsd"].analysis.greedy.bins = c

    c = ctx.low_ana_greedy_mode = Cell("str").set("binning")
    c.share(readonly=False)
    ctx.params0["low-rmsd"].analysis.greedy.mode = c

    c = ctx.low_ana_greedy_discard = Cell("int").set(50)    
    c.share(readonly=False)
    ctx.params0["low-rmsd"].analysis.greedy.discard = c

    c = ctx.low_ana_nn_bins = Cell("int").set(10)
    c.share(readonly=False)
    ctx.params0["low-rmsd"].analysis["near-native"].bins = c

    c = ctx.low_ana_nn_mode = Cell("str").set("binning")
    c.share(readonly=False)
    ctx.params0["low-rmsd"].analysis["near-native"].mode = c

    c = ctx.low_ana_nn_discard = Cell("int").set(50)    
    c.share(readonly=False)
    ctx.params0["low-rmsd"].analysis["near-native"].discard = c

    await ctx.translation()



async def load():
    from seamless.metalevel.bind_status_graph import bind_status_graph_async
    import json

    global ctx, webctx, save

    try:
        ctx
    except NameError:
        pass
    else:
        if ctx is not None:
            pr('"ctx" already exists. To reload, do "ctx = None" or "del ctx" before load()')
            return
    
    for f in (
        "web/index-CONFLICT.html",
        "web/index-CONFLICT.js",
        "web/webform-CONFLICT.txt",
    ):
        if os.path.exists(f):
            if open(f).read().rstrip("\n ") in ("", "No conflict"):
                continue
            dest = f + "-BAK"
            if os.path.exists(dest):
                os.remove(dest)            
            pr("Existing '{}' found, moving to '{}'".format(f, dest))
            shutil.move(f, dest)
    ctx = Context()
    empty_graph = await ctx._get_graph_async(copy=True)
    await define_graph(ctx)
    new_graph = await ctx._get_graph_async(copy=True)
    graph_file = "graph/" + PROJNAME + ".seamless"
    ctx.load_vault("vault")
    if new_graph != empty_graph:
        pr("*** define_graph() function detected. Not loading '{}'***\n".format(graph_file))
    else:
        pr("*** define_graph() function is empty. Loading '{}' ***\n".format(graph_file))
        graph = json.load(open(graph_file))        
        ctx.set_graph(graph, mounts=True, shares=True)
        await ctx.translation(force=True)

    status_graph = json.load(open("graph/" + PROJNAME + "-webctx.seamless"))

    webctx = await bind_status_graph_async(
        ctx, status_graph,
        mounts=True,
        shares=True
    )
    def save():
        import os, itertools, shutil

        def backup(filename):
            if not os.path.exists(filename):
                return filename
            for n in itertools.count():
                n2 = n if n else ""
                new_filename = "{}.bak{}".format(filename, n2)
                if not os.path.exists(new_filename):
                    break
            shutil.move(filename, new_filename)
            return filename

        ctx.save_graph(backup("graph/" + PROJNAME + ".seamless"))
        webctx.save_graph(backup("graph/" + PROJNAME + "-monitoring.seamless"))
        ctx.save_vault("vault")
        webctx.save_vault("vault")

    pr("""Project loaded.

    Main context is "ctx"
    Web/status context is "webctx"

    Open http://localhost:<REST server port> to see the web page
    Open http://localhost:<REST server port>/status/status.html to see the status

    Run save() to save the project
    """)
