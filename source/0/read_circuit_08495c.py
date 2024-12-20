# https://github.com/peachnuts/Multiprogramming/blob/c539b1b8d4739546909caf813f6aa8d26b6435c2/src/tools/read_circuit.py
# ======================================================================
# Copyright LIRMM (12/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                           <adrien.suau@lirmm.fr>)
#               Siyuan Niu (<siyuan.niu@lirmm.fr>)
# This software is governed by the CeCILL-B license under French law and
# abiding  by the  rules of  distribution of free software. You can use,
# modify  and/or  redistribute  the  software  under  the  terms  of the
# CeCILL-B license as circulated by CEA, CNRS and INRIA at the following
# URL "http://www.cecill.info".
#
# As a counterpart to the access to  the source code and rights to copy,
# modify and  redistribute granted  by the  license, users  are provided
# only with a limited warranty and  the software's author, the holder of
# the economic rights,  and the  successive licensors  have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using, modifying and/or  developing or reproducing  the
# software by the user in light of its specific status of free software,
# that  may mean  that it  is complicated  to manipulate,  and that also
# therefore  means that  it is reserved for  developers and  experienced
# professionals having in-depth  computer knowledge. Users are therefore
# encouraged  to load and  test  the software's  suitability as  regards
# their  requirements  in  conditions  enabling  the  security  of their
# systems  and/or  data to be  ensured and,  more generally,  to use and
# operate it in the same conditions as regards security.
#
# The fact that you  are presently reading this  means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================
import logging
from qiskit import QuantumCircuit
from pathlib import Path

logger = logging.getLogger("read_circuit")

def read_benchmark_circuit(name: str) -> QuantumCircuit:
    src_folder = Path(__file__).parent.parent
    benchmark_folder = src_folder/"benchmarks/multiprogramming/"
    return QuantumCircuit.from_qasm_file(
        benchmark_folder / f"{name}.qasm"
    )

def benchmark_circuit_path(name: str) -> str:
    src_folder = Path(__file__).parent.parent
    benchmark_folder = src_folder/f"benchmarks/multiprogramming/{name}.qasm"
    return benchmark_folder