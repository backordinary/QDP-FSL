# https://github.com/quantum-experiments/qbob/blob/6b33ea48a03e1e194dc87878b1d88395e560dff0/qbob/qiskit.py
"""Define Qiskit functionality for the OperationBuilder class."""

import re
from typing import List, Union

from qbob import qbob
from qbob.intrinsics import *
from qbob.types import *

from qiskit import *

def _get_array_indices(instruction: str) -> List[int]:
    return [int(match) for match in re.findall(r"\[(\d*)\]", instruction)]

def qbob_from_qiskit(qiskit_circuit, operation_name: str) -> qbob.OperationBuilder:
    # find the qubit allocation instruction
    qasm_instructions = qiskit_circuit.qasm().strip().split('\n')
    pc = 0
    while not qasm_instructions[pc].startswith('qreg'):
        pc += 1

    qubit_count = _get_array_indices(qasm_instructions[pc])[0]
    pc += 1

    my_qbob = qbob.OperationBuilder(operation_name)
    with my_qbob.allocate_qubits("qubits", qubit_count) as q:
        results = []
        for line in qasm_instructions[pc:]:
            tokens = line.split()
            idx = _get_array_indices(line)
            if tokens[0] == 'h':
                my_qbob += H(q[idx[0]])
            if tokens[0] == 'cx':
                my_qbob += CNOT(q[idx[0]], q[idx[1]])
            if tokens[0] == 'measure':
                results.append(M(q[idx[0]]))        
        my_qbob.returns(results)

    return my_qbob
