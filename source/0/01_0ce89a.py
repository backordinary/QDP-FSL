# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutoriale-QISKit/01-getting_started/01.py
import sys
import os
import time

# import qiskit from qiskit-sdk-py folder
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../', 'qiskit-sdk-py'))
    import Qconfig
    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
except:
    qx_config = {
        "APItoken":"da2a4002660558a35103a600bcbda7fe438cea629a6be98969ea5e367c091b6815e624bd86b6207121bd97fef79c22033318a4402eeafcbd04b021fd80f5a195",
        "url":"https://quantumexperience.ng.bluemix.net/api"
    }

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QISKitError
from qiskit import available_backends, execute, register, get_backend

# Building y<our circuit: Create it

# Create a Quantum Register called "q" with 3 qubits
qr = QuantumRegister(3)

# Create a Calassical Register called "c" with 3 qubits
cr = ClassicalRegister(3)

# Create a Quantum Circuit called involving "qr" and "cr"
circuit = QuantumCircuit(qr, cr)

# Not gate on qubit 0
circuit.x(qr[0])

# Not gate on qubit 1
circuit.x(qr[1])

# Barier to seperator the input from the circuit
circuit.barrier(qr[0])
circuit.barrier(qr[1])
circuit.barrier(qr[2])

# Toffoli gate from qubit 0, 1 to qubit 2
circuit.ccx(qr[0], qr[1], qr[2])

# CNOT (Controlled_NOT) gate from qubit 0 to qubit 1
circuit.cx(qr[0], qr[1])

# measure gate from qr to cr
circuit.measure(qr, cr)

# QASM from a program
QASM_source = circuit.qasm()
print(QASM_source)

# Visualize Circuit
# INSTALL Poppler

# from qiskit.tools.visualization import circuit_drawer
# circuit_drawer(circuit)

backend = 'local_qasm_simulator'

# Create a Quantum Program for execution

job = execute(circuit, backend)

job.status

result = job.result()

result.get_counts(circuit)

# Execute on a Real Device
register(qx_config['APItoken'], qx_config['url'])
def lowest_pending_jobs():
    """Returns the backend with the lowest pending jobs."""
    list_of_backends = available_backends(
        {'local': False, 'simulator': False})
    device_status = [get_backend(backend).status for backend in list_of_backends]
    best = min([x for x in device_status if x['available'] is True],
        key=lambda x: x['pending_jobs'])
    return best['name']

backend = lowest_pending_jobs()
print("the best backend is " + backend)

shots = 1024
max_credits = 3

job_exp = execute(circuit, backend = backend, shots= shots, max_credits= max_credits)

lapse = 0
interval = 10

while not job_exp.done:
    print('Status @ {} seconds'.format(interval * lapse))
    print(job_exp.status)
    time.sleep(interval)
    lapse += 1
print(job_exp.status)

result_real = job_exp.result()

result_real.get_counts(circuit)

jobID = job_exp.job_id
print('JOB ID: {}'.format(jobID))
jobID