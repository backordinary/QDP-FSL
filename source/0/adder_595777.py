# https://github.com/richardbowmaker/qubits/blob/c21a992f40d394b8a4660a16569073138df1fedb/Adder.py
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# source, see
# https://qiskit.org/textbook/ch-states/atoms-computation.html

def HalfAdder():

    print("\nAdder")

    # Use Aer's qasm_simulator
    simulator = QasmSimulator()

    circuit = QuantumCircuit(4, 2)
#    circuit.x(0)
    circuit.h(0)
#    circuit.x(1)
    circuit.h(1)
    circuit.barrier()
    circuit.cx(0, 2)
    circuit.cx(1, 2)
    circuit.ccx(0, 1, 3)
    circuit.barrier()
#    circuit.measure(2, 0)
#    circuit.measure(3, 1)
    circuit.measure([2,3], [0,1])
    circuit.barrier()

    print(circuit)

    # compile the circuit down to low-level QASM instructions
    # supported by the backend (not needed for simple circuits)
    compiled_circuit = transpile(circuit, simulator)

    # Execute the circuit on the qasm simulator
    job = simulator.run(compiled_circuit, shots=1000)

    # Grab results from the job
    result = job.result()
    counts = result.get_counts(circuit)
    print("\nTotal counts are:", counts)


def Adder1():

    print("\nAdder 1")

    # Use Aer's qasm_simulator
    simulator = QasmSimulator()

    qc_cnot = QuantumCircuit(2,2)
    qc_cnot.x(0)
    qc_cnot.x(1)
    qc_cnot.cx(0,1)
#    qc_cnot.measure([0,1], [0,1])
    qc_cnot.measure(0,0)
    qc_cnot.measure(1,1)

    print(qc_cnot)

    # compile the circuit down to low-level QASM instructions
    # supported by the backend (not needed for simple circuits)
    compiled_circuit = transpile(qc_cnot, simulator)

    # Execute the circuit on the qasm simulator
    job = simulator.run(compiled_circuit, shots=100)

    # Grab results from the job
    result = job.result()
    counts = result.get_counts(qc_cnot)
    print("\nTotal counts are:",counts)


