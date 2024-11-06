# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/QCP/QRNG/QRNG.py
from math import pi

from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    execute)


class QRNG:

    @staticmethod
    def random(n=1, p_0=0.5):
        'Generates an n bit string for which each bit has a p_0 probability of being 0 '
        # Probability of 1
        p_1 = 1 - p_0

        # Generate circuit
        q = QuantumRegister(n)
        c = ClassicalRegister(n)
        qc = QuantumCircuit(q, c, name="Random Generator")

        # Calculate the angle
        angle = p_1 * pi

        # Rotate 'angle' radiants around the y axis to get the desired amplitude
        qc.ry(angle, q)

        # Read the qbit
        qc.measure(q, c)

        # Get backend
        backend = Aer.get_backend("qasm_simulator")

        #
        job = execute(qc, backend, shots=1)
        counts = job.result().get_counts()
        return list(counts.keys())[0]

    @staticmethod
    def random_bell(n=1, p_0=0.5):
        '''Generates a bell state with random amplitudes and executes it 
        all the information about the state of the qubits happens inside a QuantumCircuit hence no QuantumRegister can be returned'''

        # Probability of 1
        p_1 = 1 - p_0

        # Generate cirucuit
        c = QuantumRegister(n)
        t = QuantumRegister(n)

        qc = QuantumCircuit(c, t, name="Random bell pair generator")

        angle = p_1 * pi

        qc.ry(angle, c)
        for i in range(n):
            qc.cx(c[i], t[i])

        qc.measure_all()

        # Get backend
        backend = Aer.get_backend("qasm_simulator")

        #
        job = execute(qc, backend, shots=1)
        counts = job.result().get_counts()
        s  = list(counts.keys())[0]
        mid = int(len(s)/2)
        return (s[:mid], s[mid:])

