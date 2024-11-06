# https://github.com/richardbowmaker/qubits/blob/76aa2e2d0a9b6e61c7fffda0e94500ae7b06e406/custom_gate.py
import numpy as np
import qiskit.extensions
from qiskit import *
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info.operators import Operator

def custom_gate():
    print("\ncustom gate")

    # Use Aer's qasm_simulator
    simulator = QasmSimulator()

    # Create a Quantum Circuit acting on the q register
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    circuit = QuantumCircuit(qr, cr)

    # Add a H gate on qubit 0
    # circuit.h(0)

    # hadamard gate from a matrix
    v = (1 / np.sqrt(2))
    u = np.array([[v, v], [v, -v]])
    # qu = qiskit.extensions.UnitaryGate(u)
    myh = Operator(u)
    circuit.unitary(myh, qr[0], label='mygate')


    cx = Operator([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])
    print(cx)

    h = Operator(qiskit.circuit.library.HGate())
    print(h)




    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
    #circuit.cx(0, 1)

    # Map the quantum measurement to the classical bits
    circuit.measure(qr, cr)

    # compile the circuit down to low-level QASM instructions
    # supported by the backend (not needed for simple circuits)
    compiled_circuit = transpile(circuit, simulator)

    # Execute the circuit on the qasm simulator
    job = simulator.run(compiled_circuit, shots=100)

    # Grab results from the job
    result = job.result()

    # Draw the circuit
    # circuit.draw()
    print(circuit)
    circuit.draw(output='mpl', filename='E:\_Ricks\Python\Qiskit1\circuit1.png')

    # Returns counts
    counts = result.get_counts(circuit)
    print("\nTotal count for 00 and 11 are:", counts)

