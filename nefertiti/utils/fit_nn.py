import numpy as np
import scipy.stats
import matplotlib
from io import BytesIO
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

def size_to_logN2(size):
    return np.log(size) ** 2

def logN2_to_size(logN2):
    return np.exp(np.sqrt(logN2))

def format_fn(tick_val, tick_pos):
    if tick_val < 0:
        return ''
    else:
        size = logN2_to_size(tick_val)
        if size < 100000:
            return str(int(size))
        elif np.isinf(size):
            exp = np.sqrt(tick_val) / np.log(10)                        
            s = "$1.00\cdot 10^{%d}$" % np.floor(exp)
            return s
        else:    
            s = ("$%3.2e}$" % size).replace("e+0", "e+").replace("e+", "\cdot 10^{")
            return s

def get_fitmarkers(x, y, param,binning_offset,log_binning_offset=0):
    if param["mode"] == "lowest_point":
        return x[:1].copy(), y[:1].copy()
    elif param["mode"] == "highest_point":
        return x[-1:].copy(), y[-1:].copy()
    elif param["mode"] == "midpoint":
        p = int(len(x)/2)
        return x[p:p+1].copy(), y[p:p+1].copy()
    elif param["mode"] == "binning":
        nbins = param["bins"]
        values, bins = np.histogram(x,bins=nbins)
        y = np.log(np.cumsum(values)+binning_offset) + log_binning_offset
        return bins[1:].copy(), y**2
    
def fit(x, y):
    # or: linear regression, with no first-order term
    b = 0
    a, c, _, _, _ = scipy.stats.linregress(y, x)

    p = 1/a
    q = (b*b-4*a*c)/(4*a*a)
    r = -b/(2*a)
    return a,b,c, p,q,r

def fit_nn(parameters, data, nfraglib, fraglen):

    nres = parameters["nresidues"]
    maxrmsd = 3
    xlim_margin=1.05
    threshold = data.get("threshold")

    fig = plt.figure()
    fig.set_figwidth(14*0.6)
    fig.set_figheight(12*0.6)
    ax = fig.add_subplot(111)

    ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    log_maxsize = np.log(nfraglib) * (nres - fraglen + 1)
    upper = log_maxsize**2 * 1.05
    ax.plot(
        [0, 1000], [log_maxsize**2, log_maxsize**2], 
        color="red",
        linestyle="dashed",linewidth=3
    )

    fit_points = []

    phigh1 = parameters["high-rmsd"]["computation"]
    phigh2 = parameters["high-rmsd"]["analysis"]
    high_mode = phigh2["mode"]
    high_mode = high_mode.split(",")
    if len(high_mode) == 1:
        high_fit_mode = high_show_mode = high_mode[0]
    else:
        high_fit_mode, high_show_mode = high_mode

    show_random = (high_fit_mode == "random"  or high_show_mode == "random")
    random_rmsds = data.get("random")
    if random_rmsds is None:
        show_random = False

    show_threshold = (
        high_fit_mode in ("threshold", "random") 
        or high_show_mode in ("threshold", "random") 
    )
    if threshold is None:
        show_threshold = False

    if show_threshold:
        maxrmsd = threshold
        best_of_factor = phigh1["threshold"]["best_of_factor"]
        tlog = log_maxsize - np.log(best_of_factor)
        if show_random:
            nstruc = phigh1["random"]["nstructures"]
            sample_log = np.log(best_of_factor) + np.log(nstruc) - log_maxsize
            logn0 = np.log(np.arange(len(random_rmsds))+1)
            logn = logn0 - sample_log
            x, y = random_rmsds, logn**2
            if high_fit_mode == "random":
                discard_upper = phigh2["random"].get("discard_upper")
                if discard_upper:
                    discard_upper = min(discard_upper, len(x)-1)
                    ax.plot(x[-discard_upper:], y[-discard_upper:], color="blue", linestyle="dashed",linewidth=3)
                    x = x[:-discard_upper]
                    y = y[:-discard_upper]
                discard_lower = phigh2["random"].get("discard_lower")
                if discard_lower:
                    ax.plot(x[:discard_lower], y[:discard_lower], color="blue", linestyle="dashed",linewidth=3)
                    discard_lower = min(discard_lower, len(x)-1)
                    x = x[discard_lower:]
                    y = y[discard_lower:]
                xm, ym = get_fitmarkers(
                    x, y, phigh2["random"],
                    discard_lower,
                    -sample_log
                )
                """
                size = 50 if len(xm) < 20 else 15
                ax.scatter(
                    xm, ym, color="black", 
                    marker="x", s=size,
                )
                """
                fit_points.append((xm, ym))
            
            color = "blue"
            ax.plot(x, y, color=color,linewidth=3)
        color = "blue"
        ax.scatter([threshold], [tlog**2], color=color, marker="x",s=200)
        if high_fit_mode == "threshold":
            fit_points.append(([threshold], [tlog**2]))
    

    xmin = None

    plow1 = parameters["low-rmsd"]["computation"]
    plow2 = parameters["low-rmsd"]["analysis"]
    low_mode = plow2["mode"]
    low_mode = low_mode.split(",")
    if len(low_mode) == 1:
        low_fit_mode = low_show_mode = low_mode[0]
    else:
        low_fit_mode, low_show_mode = low_mode

    show_nn_intercept = (
        low_fit_mode in ("nn", "nn_intercept") or
        low_show_mode in ("nn", "nn_intercept")
    )
    show_nn = (
        low_fit_mode == "nn" or
        low_show_mode == "nn"
    )
    if show_nn_intercept or show_nn:
        nn_rmsds = data.get("near-native")
        if nn_rmsds is None:
            show_nn = False
            show_nn_intercept = False

    if show_nn_intercept:
        nn_intercept = nn_rmsds[0]
        xmin = nn_intercept
        color = "blue" if low_fit_mode == "nn_intercept" else "green"
        color = "red"
        ax.scatter(nn_intercept, 1, color=color, marker="x", s=200,clip_on=False)
        if low_fit_mode == "nn_intercept":
            fit_points.append(([nn_intercept], [0]))


    show_greedy = (low_fit_mode == "greedy"  or low_show_mode == "greedy")
    greedy_rmsds = data.get("greedy")
    if greedy_rmsds is None:
        show_greedy = False
    if show_greedy:
        logn = np.log(np.arange(len(greedy_rmsds))+1)
        x, y = greedy_rmsds, logn**2
        discard = plow2["greedy"].get("discard")
        if discard:
            if not show_nn:
                ax.plot(x[:discard], y[:discard], color="green")
            discard = min(discard, len(x)-1)
            x = x[discard:]
            y = y[discard:]
        else:
            discard = 0
        xm, ym = get_fitmarkers(
            x, y, plow2["greedy"], discard
        )
        if len(xm) > 1:
            if not show_threshold:
                maxrmsd=xm[-1]
                xlim_margin = 1.001
                upper = ym[-1]*1.05
            a,b,c,p,q,r = fit(xm, ym)
            greedy_intercept = c
        else:
            greedy_intercept = xm[0]
        ax.plot(x, y, color="green", linewidth=10)
        ax.scatter(
            [greedy_intercept], [0], 
            color="green", marker="x", s=200,
            clip_on=False,
        )
        if low_fit_mode == "greedy":
            if high_fit_mode == "no":
                fit_points.append((xm, ym))
            else:
                fit_points.append(([greedy_intercept], [0]))

        if xmin is None or xmin > greedy_rmsds[0]:
            xmin = greedy_rmsds[0]
    
    ax.set_ylim([0, upper])
    oom = np.sqrt(upper)/np.log(10)
    best = None
    for inc in 1,2,5:
        ticks1 = oom/inc
        scale = 10**(np.floor(np.log(ticks1)/np.log(10)) - 1)
        ticks2 = int(ticks1 / scale) + 1
        if ticks2 >= 10:      
            if best is None or ticks2 < best[0]:
                best = ticks2, scale *  inc
    yticks0 = np.arange(best[0]) * best[1]
    if best[0] < 20:
        skip = int(max(10 - 0.7 * (20-best[0]), 0))
    else:
        skip = int(0.5 * best[0])
    yticks0 = yticks0[:1].tolist() + yticks0[skip:].tolist()
    yticks = [(t*np.log(10))**2 for t in yticks0]
    
    pos = ax.get_position()
    pos = [pos.x0 + 0.05, pos.y0+0.03,  pos.width, pos.height] 
    ax.set_position(pos)
    ax.set_ylabel("Cumulative ensemble size", size = "x-large")
    ax.set_yticks(yticks)
    ax.set_xlabel("RMSD (Å)", size="x-large")

    if show_nn:
        logn = np.log(np.arange(len(nn_rmsds))+1)
        x, y = nn_rmsds, logn**2
        if low_fit_mode == "nn" and len(nn_rmsds) > 1:
            discard = plow2["near-native"].get("discard")
            if discard:
                ax.plot(x[:discard], y[:discard], color="red", linestyle="dotted", clip_on=False, linewidth=4)
                discard = min(discard, len(x)-1)
                x = x[discard:]
                y = y[discard:]
            else:
                discard = 0
            xm, ym = get_fitmarkers(
                x, y, plow2["near-native"],discard
            )
            size=200 if len(xm) == 1 else (50 if len(xm) < 20 else 15)
            """
            ax.scatter(
                xm, ym, color="black", 
                marker="x", s=size,
                clip_on=False
            )
            """
            fit_points.append((xm, ym))
        color = "red"
        ax.plot(x, y, color=color,linewidth=3)

    result = {}
    
    if len(fit_points):
        fitx = np.concatenate(
            [p[0] for p in fit_points]
        )
        fity = np.concatenate(
            [p[1] for p in fit_points]
        )
        assert len(fitx) == len(fity)
        if len(fitx) > 1:
            a,b,c,p,q,r = fit(fitx, fity)
            fit_size = maxrmsd * p + q
            ax.plot(
                [c, maxrmsd], [0, fit_size],
                color="black",
                linestyle="dotted",linewidth=5
            )
            equation = {
                "a": a,
                "b": b,
                "c": c,
                "p": p,
                "q": q,
                "r": r,
            }
            text = "log(N)² = %.2f * RMSD + %.2f" % (p,q)
            text += "\n"
            text += "RMSD = %.6e * log(N)² + %.3f" % (a,c)
            equation["text"] = text
            result["equation"] = equation

    if xmin is None:
        xmin = 0
    ax.set_xlim(left=xmin/xlim_margin, right=maxrmsd*xlim_margin)

    plotobj = BytesIO()
    plt.savefig(plotobj)    
    result["plot"] = plotobj.getvalue()
    return result

if __name__ == "transformer":
    from silk import Silk
    if isinstance(data, Silk):
        data = data.unsilk
    if isinstance(parameters, Silk):
        parameters = parameters.unsilk
    result = fit_nn(parameters, data, nfraglib, fraglen)
    result["plot"] = np.array(result["plot"]) #BUG in seamless
