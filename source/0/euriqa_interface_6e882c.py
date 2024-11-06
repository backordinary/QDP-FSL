# https://github.com/Kee-Wang/euriqa-api/blob/f04ae1cc9dce22b0017ac4bdd3b2c59437c5bf5a/euriqa_interface.py
import typing
import numpy as np
import qiskit
def run_on_EURIQA(qasm_circuits:typing.List[str], num_shots:int=100, run_simulation_local=True )\
        ->typing.List[np.ndarray]:
    """

    Args:
        qasm_circuits: list of circuits where each circuit is represented by OpenQASM2.0 format.
            For best practice, please follow the guide:
            1. Use only native gate sets to generate qasm strings. In qiskit, it means gate set ['rxx', 'rx', 'ry', 'rz']
                in arbitrary angle.
            2. Supports QASM2.0. However, it is observed qiskit defines 'measurement' differently in newer versions compared
                to qiskit-terra==0.16.1 used in EURIQA. In qiskit, one can use `circuit.remove_final_measurements()`
                to remove all measurement gates before generating qasm strings.
        num_shots: Number of shots per circuit. Will be ignored if `run_simulation_local=True`
        run_simulation_local: If True, then return prob_vector simulation using the main script qiskit version without
            running actual experiment.
    Returns:
        prob_vector: list of 1D numpy arrays contains the probability of each state. For example a 2-qubit prob_vector
            of two circuits might be `[np.array([0.1, 0.1, 0.1, 0.7], np.array([0.12, 0.08, 0.1, 0.7])]

    """

    if run_simulation_local:
        circuits = [qiskit.QuantumCircuit.from_qasm_str(qasm) for qasm in qasm_circuits]
        prob_vector = [abs(qiskit.quantum_info.Statevector(cq).data)**2 for cq in circuits]
        #older versions might use abs(qiskit.quantum_info.Statevector.from_instruction(cq))**2

    else:
        raise ValueError('The connection to EURIQA hardware is not implemented yet.')

    return prob_vector
