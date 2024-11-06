# https://github.com/marcobn/Quantum/blob/0b12d7ce39d589e0810c18f7b0eb183120c6ce38/BandTheory/qctb/serializeoptimizer.py
""" Provide functions for converting aqua Optimizers to and from JSON. """

import qiskit.aqua.components.optimizers as optimizers

_META = {
    "SLSQP": {
        "class": optimizers.SLSQP,
        "extraoptions": ['tol'],
    },
    "COBYLA": {
        "class": optimizers.COBYLA,
        "extraoptions": ['tol'],
    },
    "SPSA": {
        "class": optimizers.SPSA,
        "extraoptions": ['maxiter', 'skip_calibration'], # HARD-CODE PARAMETERS
    },
    "ADAM": {
        "class": optimizers.ADAM,
        "extraoptions": [],
    }
}

def _getlabel(OPT):
    """ Find label for optimizer, or return None. """
    for label in _META:
        if OPT.__class__ == _META[label]["class"]:
            return label
    return None

def toJSON(OPT):
    """ Serialize a qiskit aqua Optimizer for later use. """
    if OPT is None: return None                 # NO OPTIMIZATION

    label = _getlabel(OPT)
    if label is None: return OPT.setting       # UNSUPPORTED OPTIMIZER
    extraoptions = {
        option: OPT.__dict__[f"_{option}"]
        for option in _META[label]["extraoptions"]
    }

    # HARD-CODE PARAMETERS FOR SPSA
    if label == "SPSA":
        for i, c in enumerate(OPT._parameters):
            extraoptions[f"c{i}"] = c

    return {
        "label": label,
        "options": OPT._options,
        "extraoptions": extraoptions,
    }

def fromJSON(OPT):
    """ Unserialize a qiskit aqua Optimizer for later use. """
    return _META[OPT["label"]]["class"](
        **OPT["options"],
        **OPT["extraoptions"],
    )
