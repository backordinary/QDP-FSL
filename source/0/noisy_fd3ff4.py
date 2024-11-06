# https://github.com/MichalReznak/quep/blob/1aad6b3afbe3f2271f2c8a7f4d1803273d39c0fc/python/noisy.py
import time
from qiskit import *
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error

# # Loose mode
# p_reset = 0.002
# p_meas = 0.001
# p_gate1 = 0.003

p_reset = 0.004
p_meas = 0.002
p_gate1 = 0.006


# QuantumError objects
error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
error_gate1 = pauli_error([('X', p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

error_reset_p = pauli_error([('Z', p_reset), ('I', 1 - p_reset)])
error_meas_p = pauli_error([('Z', p_meas), ('I', 1 - p_meas)])
error_gate1_p = pauli_error([('Z', p_gate1), ('I', 1 - p_gate1)])
error_gate2_p = error_gate1.tensor(error_gate1)

error_reset_p2 = error_reset.compose(error_reset_p)
error_meas_p2 = error_meas.compose(error_meas_p)
error_gate1_p2 = error_gate1.compose(error_gate1_p)
error_gate2_p2 = error_gate2.compose(error_gate2_p)


# Add errors to noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error_reset_p2, "reset")
noise_model.add_all_qubit_quantum_error(error_meas_p2, "measure")
noise_model.add_all_qubit_quantum_error(error_gate1_p2, ["id", "rz", "u1", "u2", "u3"])
noise_model.add_all_qubit_quantum_error(error_gate2_p2, ["cx", "zx"])


class Noisy:
    backend: AerSimulator = None
    circuits: [QuantumCircuit] = []
    meta_info: dict[str, any] = None

    def get_meta_info(self):
        return self.meta_info

    def auth(self):
        self.backend = AerSimulator(noise_model=noise_model)

    def clear_circuits(self: 'Noisy'):
        self.circuits = []

    def append_circuit(self: 'Noisy', circuit: str, t: str, log: bool):
        parsed_c = None

        if t == 'OpenQasm':
            parsed_c = QuantumCircuit.from_qasm_str(circuit)

        elif t == 'Qiskit':
            exec_res = {}
            exec(circuit, None, exec_res)
            parsed_c = exec_res["circ"]

        self.circuits.append(parsed_c)

        if log:
            print(parsed_c)

    def run_all(self: 'Noisy') -> str:
        start = time.time()
        job = execute(self.circuits, self.backend, shots=1024, memory=True, optimization_level=0)
        end = time.time()

        self.meta_info = {
            'time': end - start
        }

        return job.result().get_counts()
