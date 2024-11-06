# https://github.com/iQuHACK/2021_quanta/blob/964b24884b685cf374de2ddf377992f89996dd48/expected_outcom.py
from qiskit import Aer, execute
from qiskit.visualization.utils import _bloch_multivector_data
import numpy as np

def expected_outcome(circ):
    # Caculates the probability of measurign 0 for each qubit
    state_sim = Aer.get_backend('statevector_simulator')
    sim = execute(circ, state_sim, shots=1000)
    state_sim_result = sim.result()
    state_vec = state_sim_result.get_statevector(decimals=3)
    p0s = []
    for kk in _bloch_multivector_data(state_vec):
        if round(kk[2], 4) != 0:
            theta = np.arccos(round(kk[2], 4))
        else:
            theta = np.pi/2
        p0 = round(np.cos(theta/2)**2,3)
        p0s.append(p0)

    return p0s
