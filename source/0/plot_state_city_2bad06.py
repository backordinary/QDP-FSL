# https://github.com/jamesjcai/quantum/blob/b18436492456e62db64a8b725554e07068ef2f3e/plot_state_city.py
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.states import Statevector, partial_trace
from qiskit.visualization import plot_state_city

q0 = QuantumRegister(2, 'q0')
q1 = QuantumRegister(2, 'q1')

circuit = QuantumCircuit(q0, q1)
circuit.h(q1[1])
circuit.cswap(q1[1], q1[0], q0[1])
circuit.cx(q1[0], q0[0])
circuit.cx(q0[1], q0[0])
circuit.ccx(q0[0], q1[1], q0[1])

circuit.draw()
circuit.draw(output='mpl').show()


zero_state = Statevector.from_label('0000')
final_state = zero_state.evolve(circuit)

reduced_state = partial_trace(final_state, [0, 1])

plot_state_city(reduced_state.data).show()
