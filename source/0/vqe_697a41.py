# https://github.com/eduardsmetanin/vqe/blob/102426228b8a6caaadb9c148431631122d5dcb1f/vqe.py
import numpy as np
import qiskit
import decomposer

def run(input_hamiltonian):
    """ Runs Variational Quantum Eigensolver (VQE) algorithm.
    :returns: average (expectation) value.
    """
    decomposed_hamiltonian = decomposer.run(input_hamiltonian)

    min_avg = float("Infinity")
    for angle in np.arange(0, 2 * np.pi, 2 * np.pi / 32):
        avg = hamiltonian_average([angle], decomposed_hamiltonian)
        if avg < min_avg:
            min_avg = avg
        print(f"angle {angle:.2f} avg {avg}")

    # This commented out code uses qiskit optimizer to find ansatz parameters.
    # from scipy.optimize import minimize
    # ansatz_params = np.array([0.0])
    # optimization_precision = 1e-3
    # optimizeResult = minimize(average, ansatz_params, args=decomposed_hamiltonian, method="Powell", tol=optimization_precision)
    # min_avg = optimizeResult.fun

    return min_avg.real

def hamiltonian_average(ansatz_params, decomposed_hamiltonian):
    """ Creates and runs quantum curcuits for every Pauli term and classically adds averages for every term.
    :param ansatz_params: ansatz optimization parameters.
    :param decomposed_hamiltonian: split into weighted Pauli terms hamiltonian.
    :returns: average (expectation) value for given Hamiltonian and given ansatz parameters.
    """
    avg_I = decomposed_hamiltonian.get('II', 0) * pauli_term_average(ansatz_params, 'I')
    avg_X = decomposed_hamiltonian.get('XX', 0) * pauli_term_average(ansatz_params, 'X')
    avg_Y = decomposed_hamiltonian.get('YY', 0) * pauli_term_average(ansatz_params, 'Y')
    avg_Z = decomposed_hamiltonian.get('ZZ', 0) * pauli_term_average(ansatz_params, 'Z')
    avg = avg_I + avg_X + avg_Y + avg_Z
    return avg

def pauli_term_average(ansatz_params, measurement):
    """ Creates and runs quantum curcuit for one Pauli term.
    :param ansatz_params: ansatz optimization parameters.
    :param measurement: "I", "X", "Y" or "Z".
    :returns: average (expectation) value.
    """
    if measurement == 'I':
        return 1
    elif measurement == 'X':
        circuit = vqe_circuit(ansatz_params, 'X')
    elif measurement == 'Y':
        circuit = vqe_circuit(ansatz_params, 'Y')
    elif measurement == 'Z':
        circuit = vqe_circuit(ansatz_params, 'Z')
    else:
        raise ValueError('Given measurement is neither "I", "X", "Y" or "Z"')
    shots = 10000
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    c00 = counts.get('00', 0)
    c01 = counts.get('01', 0)
    c10 = counts.get('10', 0)
    c11 = counts.get('11', 0)
    average_value = (c00 + c11 - c01 - c10) / shots
    return average_value

def vqe_circuit(ansatz_params, measurement):
    """ Creates 2 qubit curcuit with prepared ansatz state and measurement.
    :param ansatz_params: optimization params.
    :param measurement: "X", "Y" or "Z".
    :returns: created curcuit.
    """
    if measurement not in ('X', 'Y', 'Z'):
        raise ValueError('Given measurement is neither "X", "Y" or "Z"')
    circuit = ansatz_curcuit(ansatz_params)
    if measurement == 'X':
        circuit.h(0)
        circuit.h(1)
    elif measurement == 'Y':
        circuit.u2(0, np.pi/2, 0)
        circuit.u2(0, np.pi/2, 1)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    # print(circuit.draw(output='text'))
    return circuit

def ansatz_curcuit(ansatz_params):
    """ Creates 2 qubit curcuit for ansatz I⊗X CX RZ⊗I H⊗I |00>
    :returns: created curcuit.
    """
    circuit = qiskit.QuantumCircuit(2, 2) # |00>
    circuit.h(0)                          # H⊗I
    circuit.rz(ansatz_params[0], 0)       # RZ(angle)⊗I
    circuit.cx(0, 1)                      # CX
    circuit.x(1)                          # I⊗X
    return circuit
