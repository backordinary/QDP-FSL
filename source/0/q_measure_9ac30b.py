# https://github.com/Skk-tj/q-mine/blob/1c0908c243d14d340b27af977d8a3ac49895c10f/q_measure.py
import matplotlib.pyplot as plt
import qiskit
import qiskit.visualization as qv
from enum import Enum


class MeasureResult(Enum):
    NoMine = 0
    Mine = 1


class MineState(Enum):
    NotBlown = 0
    Blew = 1


def get_q_circuit():
    # Initialize qubits
    # 1st qubit is the photon bit
    # 2nd qubit is the bomb bit
    qc = qiskit.QuantumCircuit(2)

    # Both qubits to |0>
    qc.i(0)
    qc.i(1)

    qc.barrier()

    qc.h(0)

    qc.barrier()

    # The bomb
    qc.cx(0, 1)

    qc.barrier()

    qc.h(0)

    qc.measure_all()

    return qc


def visualize_q(qc):
    qc.draw(output='mpl')
    plt.show()

    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    shots = 1024

    results = qiskit.execute(qc, backend=backend, shots=shots).result()
    answer = results.get_counts()

    qv.plot_histogram(answer, title="Quantum bomb")
    plt.show()


def get_measurement_result_for_one_shot(qc, backend='qasm_simulator'):
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    shots = 1

    results = qiskit.execute(qc, backend=backend, shots=shots).result()
    answer = results.get_counts()

    measurement_result = list(answer.keys())[0]

    return MeasureResult(int(measurement_result[0])), MineState(int(measurement_result[1]))
