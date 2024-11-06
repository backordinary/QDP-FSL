# https://github.com/chaniotakismg/quantum-experiments/blob/5490ddd16b6f36d6b5a9f6773a029e4421f7d092/basic/gates.py
#!/usr/bin/env python
# coding: utf-8
from IPython import get_ipython
from qiskit import *
from qiskit.tools.visualization import plot_bloch_multivector, plot_histogram

if __name__ == '__main__':
    circuit = QuantumCircuit(1,1)
    circuit.x(0)
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, backend=simulator).result()
    state_vector = result.get_statevector()
    print(state_vector)

    circuit.draw(output='mpl')

    plot_bloch_multivector(state_vector)

    circuit.measure([0], [0])
    backend = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend=backend, shots=1024).result()

    plot_histogram(result.get_counts())

    circuit = QuantumCircuit(1,1)
    circuit.x(0)
    unitary_simulator = Aer.get_backend('unitary_simulator')
    result_unitary = execute(circuit, backend=unitary_simulator).result()
    unitary = result_unitary.get_unitary()
    print(unitary)
