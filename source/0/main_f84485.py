# https://github.com/MiikaVuorio/qiskit2021-quantum-sim-challenge/blob/3157a584c18c70ef53a1cfa943bbe4ae4d26a0fb/main.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, IBMQ, execute, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity
from qiskit.opflow import Zero, One, I, X, Y, Z

import warnings
warnings.filterwarnings('ignore')

#IBMQ.save_account('MY_API_TOKEN')
provider = IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
jakarta = provider.get_backend('ibmq_jakarta')

sim_noisy_jakarta = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta'))
sim = QasmSimulator()


def heisenberg_matrix():
    identity = np.eye(2, 2)
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    xx_gates = np.kron(identity, np.kron(pauli_x, pauli_x)) + np.kron(pauli_x, np.kron(pauli_x, identity))
    yy_gates = np.kron(identity, np.kron(pauli_y, pauli_y)) + np.kron(pauli_y, np.kron(pauli_y, identity))
    zz_gates = np.kron(identity, np.kron(pauli_z, pauli_z)) + np.kron(pauli_z, np.kron(pauli_z, identity))

    hamiltonian = xx_gates + yy_gates + zz_gates

    return hamiltonian


def gen_xx(t):
    xx_qc = QuantumCircuit(2, name='XX')

    xx_qc.ry(np.pi / 2, [0, 1])
    xx_qc.cnot(0, 1)
    xx_qc.rz(2 * t, 1)
    xx_qc.cnot(0, 1)
    xx_qc.ry(-np.pi / 2, [0, 1])

    return xx_qc.to_instruction()


def gen_yy(t):
    yy_qc = QuantumCircuit(2, name='YY')

    yy_qc.rx(np.pi / 2, [0, 1])
    yy_qc.cnot(0, 1)
    yy_qc.rz(2 * t, 1)
    yy_qc.cnot(0, 1)
    yy_qc.rx(-np.pi / 2, [0, 1])

    return yy_qc.to_instruction()


def gen_zz(t):
    zz_qc = QuantumCircuit(2, name='ZZ')

    zz_qc.cnot(0, 1)
    zz_qc.rz(2 * t, 1)
    zz_qc.cnot(0, 1)

    return zz_qc.to_instruction()


def gen_cc(t):
    cc_qc = QuantumCircuit(2, name='CC')

    cc_qc.cnot(0, 1)
    cc_qc.rx(2 * t - np.pi / 2, 0)
    cc_qc.rz(2 * t, 1)
    cc_qc.h(0)
    cc_qc.cnot(0, 1)
    cc_qc.h(0)
    cc_qc.rz(-2 * t, 1)
    cc_qc.cnot(0, 1)
    cc_qc.rx(np.pi / 2, 0)
    cc_qc.rx(-np.pi / 2, 1)

    return cc_qc.to_instruction()


def og_trot_step(t):
    trot_qc = QuantumCircuit(3, name='Trot')
    trot_qc.append(gen_xx(t), [0, 0 + 1])
    trot_qc.append(gen_yy(t), [0, 0 + 1])
    trot_qc.append(gen_zz(t), [0, 0 + 1])
    trot_qc.append(gen_xx(t), [1, 1 + 1])
    trot_qc.append(gen_yy(t), [1, 1 + 1])
    trot_qc.append(gen_zz(t), [1, 1 + 1])

    return trot_qc.to_instruction()

def fresh_trot_step(t):
    trot_qc = QuantumCircuit(3, name='Trot')
    trot_qc.append(gen_cc(t), [0, 0 + 1])
    trot_qc.append(gen_cc(t), [1, 1 + 1])

    return trot_qc.to_instruction()


def gen_trot_circ(steps, trot_decomp):
    t = np.pi/steps
    final_circ = QuantumCircuit(7)

    # Prepare |110> for qubits 1, 3, 5
    final_circ.x([3, 5])

    for step in range(steps):
        final_circ.append(trot_decomp(t), [1, 3, 5])

    return state_tomography_circuits(final_circ, [1, 3, 5])


def run_circuits(n_points, backend, shots, trot_decomp):
    q_jobs = []
    for t_steps in range(n_points):
        st_qcs = gen_trot_circ(t_steps + 4, trot_decomp)
        job = execute(st_qcs, backend, shots=shots)
        q_jobs.append(job)

    return q_jobs, st_qcs


def tomographies(q_jobs, st_qcs):
    target_state = (One ^ One ^ Zero).to_matrix()
    fids = []

    for job in q_jobs:
        tomo_fitter = StateTomographyFitter(job.result(), st_qcs)
        rho_fit = tomo_fitter.fit(method='lstsq')
        fid = state_fidelity(rho_fit, target_state)
        fids.append(fid)

    return fids


def plotter(data, titl, num):
    xs = []
    for n in range(len(data)):
        xs.append(n + 4)

    plt.figure(num)
    plt.plot(xs, data)
    plt.xlabel('Number of trotter steps')
    plt.ylabel('Fidelity of result')
    plt.title(titl)


#print(heisenberg_matrix())

n_of_points = 24
ideal_q_jobs, i_st_qcs = run_circuits(n_of_points, sim, 1024, fresh_trot_step)
noisy_q_jobs, n_st_qcs = run_circuits(n_of_points, sim_noisy_jakarta, 1024, fresh_trot_step)

ideal_data = tomographies(ideal_q_jobs, i_st_qcs)
noisy_data = tomographies(noisy_q_jobs, n_st_qcs)

print(ideal_data)
print(noisy_data)

plotter(ideal_data, 'Fidelity of result per number of trotter steps in an error free simulation', 1)
plotter(noisy_data, 'Fidelity of result per number of trotter steps in a noisy simulation', 2)
plt.show()
