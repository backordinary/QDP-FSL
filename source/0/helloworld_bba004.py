# https://github.com/richardbowmaker/qubits/blob/c21a992f40d394b8a4660a16569073138df1fedb/HelloWorld.py

from qiskit import QuantumCircuit, transpile, assemble, Aer, execute, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram



def HelloWorld():

    print("\nHello world")

    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    circuit = QuantumCircuit(qr, cr)

    circuit.h(qr[0])
    circuit.cx(qr[0], qr[1])

    circuit.measure(qr, cr)

#    simulator = QasmSimulator()
    simulator = Aer.get_backend('qasm_simulator')
    simulator = Aer.get_backend('statevector_simulator')

    result = execute(circuit, simulator, shots=100).result()
    statevector = result.get_statevector()

#    plot_histogram(result.get_counts(circuit))
#    circuit.draw(output='mpl', filename='E:\_Ricks\Python\Qiskit1\circuit2.png')

    print(circuit)


    print(statevector)







