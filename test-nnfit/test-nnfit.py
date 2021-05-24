import numpy as np

NAME = ""

def load_greedy(name, poolsize):
    f = "data/greedy-%s-poolsize-%d-rmsd.npy"
    return np.load(f % (name, poolsize))


def load_kbest(name, k):
    f = "data/kbest-%s-k-100000-rmsd.npy"
    result = np.load(f % name)
    return result[:k]


def load_randombest(name):
    f = "data/randombest-%s-rmsd.npy"
    rmsds = np.load(f % name)
    f = "data/randombest-%s-threshold.txt"
    threshold = float(open(f % name).read())
    return rmsds, threshold

from seamless.highlevel import Context, Cell, Transformer
ctx = Context()
ctx.testdata0 = Cell("cson")
ctx.testdata0.mount("test-nnfit.cson")
ctx.testdata = Cell("plain")
ctx.testdata = ctx.testdata0
ctx.testdata.mount("test-nnfit.json", "w")
ctx.parameters = Cell("plain").mount("params.json", authority="cell")#.set({})
ctx.data = Cell("mixed")
ctx.fit_nn = Transformer()
ctx.fit_nn.code.mount("fit_nn.py")
ctx.fit_nn.parameters = ctx.parameters
ctx.fit_nn.data = ctx.data
ctx.fit_nn.nfraglib = 111
ctx.fit_nn.fraglen = 4
ctx.fitted_nn = Cell()
ctx.fitted_nn = ctx.fit_nn
ctx.plot = Cell("bytes")
ctx.plot.mount("plot.png", "w")
ctx.plot = ctx.fitted_nn["plot"]
ctx.equation = Cell("plain")
ctx.equation = ctx.fitted_nn["equation"]
ctx.equation.mount("equation.json", "w")
ctx.nn_cost = Transformer()
ctx.nn_cost.code.mount("nn_cost.py", "r")
ctx.nn_cost.parameters = ctx.parameters
ctx.validated_parameters = ctx.nn_cost 
ctx.compute()

def load_parameters(name):
    global NAME
    NAME = name
    ctx.data = None
    testdata = ctx.testdata.value
    if testdata is None:
        print("testdata is None")
        return
    p = [pp for pp in testdata if pp["name"] == name][0]
    parameters = p.copy()
    high = {}
    high["mode"] = "random"
    high["mode_@1"] = "no"
    high["mode_@2"] = "no,threshold"
    high["mode_@3"] = "no,random"
    high["mode_@4"] = "threshold"
    high["mode_@5"] = "threshold,random"
    high["mode_@6"] = "random"
    high["random"] = {
        "discard_upper": 0,
        "discard_lower": 0,
        "mode": "binning",
        "mode_@1": "binning",
        "mode_@2": "lowest_point",
        "mode_@3": "midpoint",
        "mode_@4": "highest_point",
        "bins": 50
    }
    parameters["high-rmsd"]["analysis"] = high

    low = {}
    has_nn = "near-native" in parameters["low-rmsd"]["computation"]
    c = parameters["low-rmsd"]["computation"]
    c["greedy"]["poolsize"] = 500
    c["greedy"]["poolsize_@1"] = 100
    c["greedy"]["poolsize_@2"] = 200
    c["greedy"]["poolsize_@3"] = 500
    c["greedy"]["poolsize_@4"] = 1000
    if has_nn:
        c["near-native"]["k"] = 100000
        c["near-native"]["k_@1"] = 1
        c["near-native"]["k_@2"] = 10000
        c["near-native"]["k_@3"] = 100000
    low["mode"] = "nn"
    low["mode_@1"] = "no"
    low["mode_@2"] = "no,greedy"
    if has_nn:
        low["mode_@3"] = "no,nn_intercept"
        low["mode_@4"] = "greedy"
    else:
        low["mode_@3"] = "greedy"
    if has_nn:
        low["mode_@5"] = "greedy,nn_intercept"
        low["mode_@6"] = "greedy,nn"
        low["mode_@7"] = "nn_intercept"
        low["mode_@8"] = "nn_intercept,greedy"
        low["mode_@9"] = "nn_intercept,nn"
        low["mode_@A"] = "nn"
        low["mode_@B"] = "nn,greedy"
    low["greedy"] = {
        "discard": 0,
        "mode": "binning",
        "mode_@1": "binning",
        "mode_@2": "lowest_point",
        "bins": 50
    }
    if has_nn:
        low["near-native"] = {
            "discard": 0,
            "mode": "binning",
            "mode_@1": "binning",
            "mode_@2": "lowest_point",
            "mode_@3": "midpoint",
            "mode_@4": "highest_point",
            "bins": 50
        }

    parameters["low-rmsd"]["analysis"] = low
    ctx.parameters.set(parameters)


def load_data(greedy=True, random=True, threshold=True, nn_intercept=True, nn=True):
    parameters = ctx.parameters.value
    data = {}
    if greedy:
        d = load_greedy(
            NAME, 
            parameters["low-rmsd"]["computation"]["greedy"]["poolsize"]
        )
        data["greedy"] = d
    if nn_intercept or nn:
        if nn:
            k = parameters["low-rmsd"]["computation"]["near-native"]["k"]
        else:
            k = 1
        d = load_kbest(NAME, k)
        data["near-native"] = d
    if random:
        assert threshold
    if threshold:
        rmsds, threshold = load_randombest(NAME)
        data["threshold"] = threshold
        if random:
            data["random"] = rmsds
    ctx.data = data

load_parameters("octa1")
ctx.compute(0.1)
load_data()        