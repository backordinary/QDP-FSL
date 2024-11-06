# https://github.com/quantumyatra/quantum_computing/blob/5723b8639ffa6130f229d42141d1a108b7dc5fd3/utils/utils.py
#!/usr/bin/env python3

import qiskit
from qiskit import Aer, execute

def  get_statevector(qc, counts=False):
    backend = Aer.get_backend('statevector_simulator')
    res  = execute(qc, backend=backend).result()
    final_state = res.get_statevector()
    counts = res.get_counts()
    if counts:
        return (final_state, counts)
    return final_state
