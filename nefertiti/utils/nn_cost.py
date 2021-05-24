from seamless.core.transformation import SeamlessTransformationError


def nn_cost(parameters):
    """Returns if the cost is too high, plus a reason"""
    nres = parameters["nresidues"]
    
    if nres > 1000:
        return True, "Protein is too big"

    phigh1 = parameters["high-rmsd"]["computation"]
    phigh2 = parameters["high-rmsd"]["analysis"]
    high_mode = phigh2["mode"]
    high_mode = high_mode.split(",")
    if len(high_mode) == 1:
        high_fit_mode = high_show_mode = high_mode[0]
    else:
        high_fit_mode, high_show_mode = high_mode


    if nres > 250:
        if high_fit_mode == "threshold" or high_show_mode == "threshold":
            return True, "Too large for threshold calculation" #  until memory issue solved
    if nres > 120:
        if high_fit_mode == "random" or high_show_mode == "random":
            return True, "Too large for random best" #  until memory issue solved
    
    if high_fit_mode in ("threshold", "random") or high_show_mode in ("threshold", "random"):
        nstruc = phigh1["random"]["nstructures"]
        if nstruc > 1000:
            return True, "More than 1000 random structures" # for now#
        best_of_factor = phigh1["threshold"]["best_of_factor"]
        nsample1 = best_of_factor * phigh1["threshold"]["redundancy"]   
        if nres > 12:
            if nsample1 > 20000:
                return True, "Too many samples for threshold detection"
        else:
            if nsample1 > 200000:
                return True, "Too many samples for threshold detection"
        if high_fit_mode == "random" or high_show_mode == "random":
            nsample2 = best_of_factor * nstruc
            if nres > 12:
                if nsample2 > 100000:
                    return True, "Too many samples for random best"
            else:
                if nsample2 > 1000000:
                    return True, "Too many samples for random best"

    plow1 = parameters["low-rmsd"]["computation"]
    plow2 = parameters["low-rmsd"]["analysis"]
    low_mode = plow2["mode"]
    low_mode = low_mode.split(",")
    if len(low_mode) == 1:
        low_fit_mode = low_show_mode = low_mode[0]
    else:
        low_fit_mode, low_show_mode = low_mode
    
    if low_fit_mode == "greedy" or low_show_mode == "greedy":
        poolsize = plow1["greedy"]["poolsize"]
        if poolsize > 1000:
            return True, "Greedy poolsize too large"
        if nres > 500 and poolsize > 100:
            return True, "Greedy poolsize too large for this protein size"
        if nres > 100 and poolsize > 200:
            return True, "Greedy poolsize too large for this protein size"
        if nres > 12 and poolsize > 500:
            return True, "Greedy poolsize too large for this protein/peptide size"

    if low_fit_mode == "nn" or low_show_mode == "nn":
        k = plow1["near-native"]["k"]
        if nres > 20:
            return True, "Maximum peptide size 20 for near-native ensemble calculation"
        if nres > 12 and k > 1:
            return True, "Only k=1 for peptides above size 12 for near-native ensemble calculation"
        if nres > 8 and k > 10000:
            return True, "k too big for near-native ensemble"
        if k > 100000:
            return True, "k too big for near-native ensemble"

    return False, None

if __name__ == "transformer":
    from silk import Silk
    if isinstance(parameters, Silk):
        parameters = parameters.unsilk
    has_err, reason = nn_cost(parameters)
    if has_err:
        raise SeamlessTransformationError(reason)
    result = parameters
