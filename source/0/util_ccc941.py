# https://github.com/latticesurgery-com/lattice-surgery-compiler/blob/3c2039d8be2416c2d7984424f368de9f636c677a/debug/util.py
from typing import *
import qiskit.opflow as qkop


def nice_print_dict_of_dict_states(ds : Dict[int, qkop.DictStateFn]) -> str:
    out = "{"
    for k ,v in ds.items():
        out += "\n\t" + str(k) + " : "
        out +=        str(v.primitive) + " ,"
    return out + "}"
