# https://github.com/ATourkine/QOSF_QMP/blob/a00dbd783f2ab0d1cadd17f5c109b2cf54492de3/NoiseModelTask2.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit import Aer
from qiskit.providers.aer.noise import NoiseModel, pauli_error

def buildNoiseModel(errorProba, probaType):
    noise_bit_flip = NoiseModel()
    error = pauli_error([('X', probaType * errorProba), ('I', 1 - errorProba), ('Z', (1 - probaType) * errorProba)])
    noise_bit_flip.add_all_qubit_quantum_error(error, ["id"])
    return noise_bit_flip

qb = QuantumRegister(2, 'q')
c = ClassicalRegister(2)
circ = QuantumCircuit(qb, c)

# Make a circuit
circ.h(qb)
# Introducing error:
circ.id(qb)

circ.cx(qb[0], qb[1])
circ.h(qb)
# circ.measure(qb[1], c[1])
circ.measure(qb[0], c[0])

# Perform a noise simulation
noise_bit_flip = buildNoiseModel(0.3, 0.7)
result = execute(circ, Aer.get_backend('qasm_simulator'),
                 # coupling_map=coupling_map,
                 basis_gates=noise_bit_flip.basis_gates,
                 noise_model=noise_bit_flip,
                 shots = 10000).result()
counts = result.get_counts(0)
print(counts)
