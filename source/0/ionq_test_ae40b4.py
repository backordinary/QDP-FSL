# https://github.com/sybarits/quantum-computing/blob/6d5c872ed64cc969aa5611e3a291811561f6415c/q-play/ionq-test.py
# import utils
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
# import gates
from qiskit_ionq import GPIGate, GPI2Gate, MSGate


# initialize a quantum circuit
circuit = QuantumCircuit(2, 2)
# add gates
circuit.append(MSGate(0, 0), [0, 1])
circuit.append(GPIGate(0), [0])
circuit.append(GPI2Gate(1), [1])
circuit.measure([0, 1], [0, 1])
print(circuit.draw())


provider = IonQProvider("my key")
native_simulator = provider.get_backend("ionq_simulator", gateset="native")

print(provider.backends())

job = native_simulator.run(circuit)
job.get_probabilities()
