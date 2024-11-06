# https://github.com/ankurbhambri/Quantum-Computing/blob/546203048374c66f2e68ce14fbf0b0c2232705d3/phase_bloch_sphere.py
from qiskit import *
from qiskit.tools.visualization import plot_bloch_multivector
from qiskit.visualization import plot_histogram
import math

qasm_simulator = Aer.get_backend("qasm_simulator")
statevector_simulator = Aer.get_backend("statevector_simulator")


def run_on_simulators(circuit):
    statevec_job = execute(circuit, backend=statevector_simulator)
    statevec = statevec_job.result().get_statevector()
    num_quibits = circuit.num_qubits
    circuit.measure(
        [i for i in range(num_quibits)], [i for i in range(num_quibits)]
    )
    qasm_job = execute(circuit, backend=qasm_simulator, shots=1024).result()
    counts = qasm_job.get_counts()
    return statevec, counts


circuits = QuantumCircuit(2, 2)
statevec, counts = run_on_simulators(circuits)
plot_bloch_multivector(statevec)
