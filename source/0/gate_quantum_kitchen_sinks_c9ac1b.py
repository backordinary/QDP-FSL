# https://github.com/shaunakpp/quantum_kitchen_sinks/blob/300adb1a2e80b93749b42492bc7dfda7d459cfbe/gate_quantum_kitchen_sinks.py
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, Aer, execute
import numpy as np
from quantum_kitchen_sink import QuantumKitchenSink

class GateQuantumKitchenSink(QuantumKitchenSink):
    def run_quantum_program(self, theta):
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(qr, cr)

        qc.rx(theta[0], qr[0])
        qc.rx(theta[1], qr[1])
        qc.cx(qr[0], qr[1])

        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        simulator = Aer.get_backend('qasm_simulator')
        job = execute(qc, simulator, shots=self.shots)

        result = job.result()

        counts = result.get_counts(qc)

        means = [0.0, 0.0]
        for state in counts:
            if list(state)[0] == '1':
                means[0] += counts[state]
            if list(state)[1] == '1':
                means[1] += counts[state]

        means = np.array(means)/self.shots
        return means

