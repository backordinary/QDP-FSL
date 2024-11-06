# https://github.com/benc2/BachelorProjectVQA/blob/d359b58a5df86085990cd2171eb142df368caf62/utils.py
import numpy as np
from qiskit import *
import matplotlib.pyplot as plt
from collections import defaultdict


def gen_state_circuit(theta):  # generates state (|0⟩ + e^iθ |1⟩) / √2
    circ = QuantumCircuit(1)
    circ.h(0)
    circ.p(theta, 0)  # p gate (aka u1) adds phase factor to |1⟩ state
    return circ


def measuring_circuit_ghz(axis, N):
    q = QuantumRegister(N)
    c = ClassicalRegister(1)
    circ = QuantumCircuit(q, c)
    if axis == 'y':
        circ.sdg(0)
    elif axis != 'x':
        raise ValueError("Invalid axis")
    if N == 1:
        circ.h(0)
        circ.measure(0, c)
    else:
        for i in range(N):
            circ.h(i)
            circ.measure(i, c)
            circ.z(N-1).c_if(c, 1)
    return circ


def measuring_circuit_1qubit(axis, N=1):  # meas
    circ = QuantumCircuit(N)
    if axis == 'y':
        circ.sdg(0)
    elif axis != 'x':
        raise ValueError("Invalid axis")
    circ.h(0)
    circ.measure_all()
    return circ


def estimate_theta(state_circuit, shots=1024, backend_sim=Aer.get_backend('qasm_simulator')):
    # estimates theta in range [-π, π]
    N = state_circuit.num_qubits
    shots //= 2   # divide by 2 to spread across x and y
    # simulate results for cos(θ)
    # measuring_circuit_ghz('x', N).draw('mpl'); plt.show()
    circ = state_circuit.compose(measuring_circuit_ghz('x', N))  # combine state-generating- & measurement circuit
    circ = transpile(circ, backend_sim)
    counts = defaultdict(int, execute(circ, backend_sim, shots=shots).result().get_counts())
    x = 2 * counts['0'] / shots - 1

    # simulate results for sin(θ)
    circ = state_circuit.compose(measuring_circuit_ghz('y', N))
    circ = transpile(circ, backend_sim)
    counts = defaultdict(int, execute(circ, backend_sim, shots=shots).result().get_counts())
    y = 2 * counts['0'] / shots - 1

    theta_estimate = np.arctan2(y, x)

    # error propagation
    norm = np.sqrt(x ** 2 + y ** 2)
    sd_x = np.sqrt((1-x**2)/shots)
    sd_y = np.sqrt((1-y**2)/shots)
    sd_theta = np.sqrt((y/norm * sd_x)**2 + (x/norm * sd_y)**2)
    return theta_estimate,  sd_theta



def print_summary(est, sd, true_theta=None, digits=3):
    print(f"θ = {est / np.pi:.{digits}f}π ± {sd / np.pi:.{digits}f}π")
    if true_theta is not None:
        print("True value in 1σ confidence interval?", est - sd < true_theta < est + sd)


def simulate_statevector(circuit):
    """
    Simulates statevector from circuit
    Args:
        circuit: qiskit QuantumCircuit

    Returns:
        complex numpy array: statevector
    """
    print(circuit)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    return result.get_statevector()


def plot_probabilities(statevector, nonzero=False, **kwargs):
    """
    Plots probabilities from state vector
    Args:
        statevector: statevector (numpy array) of length 2**n for some number of qubits n containing the coefficients
                        of the qubit states (from low to high, with MSB on the left. This is for the labelling.)
        nonzero: whether to include states with probability 0 (<1e-15 are thrown out)
        **kwargs: passed on to plt.bar
    """
    probabilities = np.abs(statevector) ** 2
    n = int(np.log2(len(statevector)))
    N = len(statevector)
    state_names = np.array([f"|{i:0>{n}b}⟩" for i in range(N)])  # create labels: get binary repr, pad with 0s
    if nonzero:  # filter out probabilities of (very close to) zero
        keep_indices = probabilities > 1e-15
        state_names = state_names[keep_indices]
        probabilities = probabilities[keep_indices]
    plt.bar(state_names, probabilities, **kwargs)
    plt.show()


def str_coefficient(z, polar=False, digits=3):
    """
    Turns numpy complex number into printable string
    Args:
        z: complex number
        polar: Optional; whether or not to display z in complex notation. If number is purely real/imaginary,
                polar notation is not used
        digits: digits after the decimal point

    Returns:
        string representing z
    """
    z = complex(z)
    if polar:
        rounded_abs = round(np.abs(z), digits)
        if np.isclose(np.angle(z), 0):
            return f"{rounded_abs}"
        elif np.isclose(np.angle(z), np.pi):
            return f"-{rounded_abs}"
        elif np.isclose(np.angle(z), np.pi/2):
            return f"{rounded_abs}i"
        elif np.isclose(np.angle(z), -np.pi/2):
            return f"-{rounded_abs}i"
        return f"{rounded_abs} e^({round(np.angle(z) / np.pi, digits)}πi)"
    else:
        if np.isclose(np.imag(z), 0):
            return str(round(np.real(z), digits))
        elif np.isclose(np.real(z), 0):
            return str(round(np.imag(z))) + 'i'
        return str(f"{z:.3}")


def str_statevector(statevector, polar=False, precision=3, sympy=False, newlines=False):
    N = round(np.log2(len(statevector)))
    state_names = np.array([f"|{i:0>{N}b}⟩" for i in range(2**N)])
    # result = ""
    if newlines:
        spacing = "\n"
    else:
        spacing = ""
    if sympy:
        terms = [f"({repr(x)}) {state}" for x, state in zip(statevector, state_names) if x != 0]
    else:
        terms = [str_coefficient(coeff, polar=polar, digits=precision) + ' ' + state
                 for coeff, state in zip(statevector, state_names) if np.abs(coeff) > 1e-15]

    return f' + {spacing}'.join(terms)


def percentage_in_1sd(estimator, true_param, shots=1000):
    score = 0
    for i in range(shots):
        t, s = estimator()
        if t - s < true_param < t + s:
            score += 1
    return score/shots


