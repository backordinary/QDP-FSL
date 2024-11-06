# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/test_inverse.py
from qiskit import (IBMQ, Aer, ClassicalRegister, QuantumCircuit,
                    QuantumRegister, execute)
from qiskit.backends.ibmq import least_busy
from qiskit.tools.visualization import circuit_drawer

id_string = "test_inverse"

qr = QuantumRegister(1)
cr = ClassicalRegister(1)

circuit = QuantumCircuit(qr, cr)
circuit.h(qr[0])
circuit.h(qr[0])
circuit.measure(qr, cr)
circuit_drawer(circuit, filename="imgs/" + id_string + ".png")

shots = 4096
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(circuit, shots=shots, backend=backend_sim)
result_sim = job_sim.result()
# Show the results
print("simulation: ", result_sim)
print(result_sim.get_counts(circuit))

IBMQ.load_accounts()
large_enough_devices = IBMQ.backends(
    filters=
    lambda x: x.configuration()['n_qubits'] > 1 and not x.configuration()['simulator']
)
backend_real = least_busy(large_enough_devices)
print("The best backend is " + backend_real.name())

max_credits = 3
job_real = execute(
    circuit, backend=backend_real, shots=shots, max_credits=max_credits)
result_real = (job_real.result())
print("real: ", result_real)
print(result_real.get_counts(circuit))
