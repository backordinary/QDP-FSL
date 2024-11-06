# https://github.com/payvint/quantum-automata/blob/b4eab4fe8976c336c47c408435aec86c827d82e7/core/Registry.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import random_statevector, Statevector
# from Automaton import Gate

class Registry:
    def __init__(self, number):
        self.quantum_registry = QuantumRegister(number)
        self.classical_registry = ClassicalRegister(number)
        self.quantum_circuit = QuantumCircuit(self.quantum_registry, self.classical_registry, name='circuit')
        self.quantum_circuit.initialize(random_statevector(2 ** number).data, self.quantum_registry)
        # initialize different simulator??
        self.simulator = QasmSimulator()
    
    def measure(self):
        # add posibility to measure several qubits not the all register
        self.quantum_circuit.measure(self.quantum_registry, self.classical_registry)
        compiled_circuit = transpile(self.quantum_circuit, self.simulator)
        job = self.simulator.run(compiled_circuit, shots=1024)
        res = job.result()
        return self.__calculateMeasurment(res.get_counts(self.quantum_circuit))

    def applyGate(self, gate):
        # currently switch case, add gate from library in Gate class in the future
        if gate.getGate() == 'X':
            self.quantum_circuit.x(self.quantum_registry[gate.getApplyPos()])
            return self.measure()
    
    def __calculateMeasurment(self, result):
        maxValue = 0
        maxKey = 0
        print("Measure:", result)
        for key, value in result.items():
            if value > maxValue:
                maxKey = key
                maxValue = value
        return maxKey
    
    def draw(self):
        print(self.quantum_circuit.draw())