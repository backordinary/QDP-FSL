# https://github.com/DanilShayakhmetov/graduate_project/blob/25ee4b26e68ed6c587a02d607cdf69ebc216da3e/test.py
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor


from qiskit import IBMQ
from sympy.physics.continuum_mechanics.beam import matplotlib

IBMQ.save_account('a5718e05677118aa192e1baac686bb764972b28ff18f673b8bffbb6834fe874227f0c794da513898589645b757366711e1f3681e70ec2f3c74197df9bc0f8ef7')

IBMQ.load_accounts()
IBMQ.backends()


device = IBMQ.get_backend('ibmq_16_melbourne')
properties = device.properties()
coupling_map = device.configuration().coupling_map



# Construct quantum circuit
# qr = QuantumRegister(3, 'qr')
# cr = ClassicalRegister(3, 'cr')
# circ = QuantumCircuit(qr, cr)
# circ.h(qr[0])
# circ.cx(qr[0], qr[1])
# circ.cx(qr[1], qr[2])
# circ.measure(qr, cr)


q1 = QuantumRegister(9)
c1 = ClassicalRegister(9)
circ = QuantumCircuit(q1, c1)



# qc1.x(q1[0])
circ.x(q1[1])
#

# qc1.x(q1[5])

circ.x(q1[6])



circ.h(q1[1])
circ.h(q1[6])

circ.x(q1[1])
circ.ccx(q1[0], q1[1], q1[2])
circ.x(q1[0])
circ.x(q1[1])
circ.ccx(q1[0], q1[1], q1[3])
circ.x(q1[0])

circ.x(q1[6])
circ.ccx(q1[5], q1[6], q1[7])
circ.x(q1[5])
circ.x(q1[6])
circ.ccx(q1[5], q1[6], q1[8])
circ.x(q1[5])

circ.ccx(q1[2], q1[3], q1[4])
circ.ccx(q1[7], q1[4], q1[2])
circ.ccx(q1[8], q1[4], q1[3])



circ.measure(q1[2], c1[2])
circ.measure(q1[3], c1[3])





# Select the QasmSimulator from the Aer provider
simulator = Aer.get_backend('qasm_simulator')

# Execute and get counts
result = execute(circ, simulator).result()
counts = result.get_counts(circ)
plot_histogram(counts, title='Ideal counts for 3-qubit GHZ state')


gate_times = [
    ('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
    ('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
    ('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
    ('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
    ('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
    ('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
    ('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)
]

# Construct the noise model from backend properties
# and custom gate times
noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)
print(noise_model)



# Get the basis gates for the noise model
basis_gates = noise_model.basis_gates

# Select the QasmSimulator from the Aer provider
simulator = Aer.get_backend('qasm_simulator')

# Execute noisy simulation and get counts
result_noise = execute(circ, simulator,
                       noise_model=noise_model,
                       coupling_map=coupling_map,
                       basis_gates=basis_gates).result()
counts_noise = result_noise.get_counts(circ)
# plot_histogram(counts_noise, title="Grover noise model")
print(counts_noise)

# print(result_noise.get_counts(circ))