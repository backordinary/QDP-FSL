# https://github.com/flyingperson23/Quantum/blob/2c565c22e29986ff4e2b107b3215bb5200a17ff2/VQE.py

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, transpile, assemble

backend = Aer.get_backend("qasm_simulator")
NUM_SHOTS = 10000

# qubits are ordered as (site 0 up, site 0 down, site 1 up, site 1 down)

# t and mu are both subtracted from energy, so they're positive here
U = 2
#mu = U / 2
mu = 0
t = 1

# given the dict structure in which each state is a key and the occurrences of that state is the value, returns the
# probability that, for a random value, the corresponding state will have '1' for both sites i and j
def get_prob_i_and_j(data: dict, i, j):
    prob = 0.0
    for key in data.keys():
        if key[i] == '1' and key[j] == '1':
        #if key[i] == '1' and key[j] == '1' and key != "1110" and key != "0111" and key != "1111":
            prob += data.get(key)
    return prob / sum(data.values())

def get_prob_hopping(data: dict, i, j):
    prob = 0.0
    prob2 = 0.0
    for key in data.keys():
        if key == "1000" or key == "0100" or key == "0010" or key == "0001" or True:
          if key[i] == '1' and key[j] == '0':
            prob += data.get(key)
          if key[i] == '0' and key[j] == '1':
            prob2 += data.get(key)
    return (prob2-prob) / sum(data.values())

def apply_ansatz(qc: QuantumCircuit, qr: QuantumRegister, params):
    qc.u3(params[0], params[1], params[2], qr[0])
    qc.u3(params[3], params[4], params[5], qr[1])
    qc.u3(params[6], params[7], params[8], qr[2])
    qc.u3(params[9], params[10], params[11], qr[3])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[2], qr[3])
    qc.cx(qr[1], qr[2])
    qc.u3(params[0], params[1], params[2], qr[0])
    qc.u3(params[3], params[4], params[5], qr[1])
    qc.u3(params[6], params[7], params[8], qr[2])
    qc.u3(params[9], params[10], params[11], qr[3])

# no transformation
def get_onsite_terms(params):
    qr = QuantumRegister(4, name="q")
    cr = ClassicalRegister(4, name="c")
    qc = QuantumCircuit(qr, cr)
    apply_ansatz(qc, qr, params)
    qc.measure(qr, cr)
    return qc

# transformation to 1/2 (X_i X_j + Y_i Y_j) basis as described in https://arxiv.org/abs/1912.06007, fig 5
def get_hopping_term(params, i, j):
    qr = QuantumRegister(4, name="q")
    cr = ClassicalRegister(4, name="c")
    qc = QuantumCircuit(qr, cr)
    apply_ansatz(qc, qr, params)
    qc.cx(qr[i], qr[j])
    qc.ch(qr[j], qr[i])
    qc.cx(qr[i], qr[j])
    qc.measure(qr, cr)
    return qc


def get_hopping_energy(params):
    energy = 0.0

    # each value of ij is used to calculate the hopping energy from site ij[0] to site ij[1]
    for ij in [(0, 2), (1, 3), (2, 0), (3, 1)]:
        qc = get_hopping_term(params, ij[0], ij[1])
        t_qc = transpile(qc, backend)
        qobj = assemble(t_qc, shots=NUM_SHOTS)
        results = backend.run(qobj).result().get_counts(qc)
        energy -= t * get_prob_hopping(results, ij[0], ij[1])
    return energy

def get_onsite_and_chemical_energy(params):
    qc = get_onsite_terms(params)
    t_qc = transpile(qc, backend)
    qobj = assemble(t_qc, shots=NUM_SHOTS)
    results = backend.run(qobj).result().get_counts(qc)
    energy = 0.0
    for ij in [(0, 1), (2, 3)]:
        energy += U * get_prob_i_and_j(results, ij[0], ij[1])

    #for site in range(4):
    #    energy -= mu * get_prob_i_and_j(results, site, site)

    return energy

def objective_function(params):
    energy = get_onsite_and_chemical_energy(params) + get_hopping_energy(params)
    print(str(params)+" "+str(energy))
    return energy

from qiskit.aqua.components.optimizers import SPSA

optimizer = SPSA(maxiter=500)

params = np.random.rand(12)
ret = optimizer.optimize(num_vars=12, objective_function=objective_function, initial_point=params)

print(ret[0])
