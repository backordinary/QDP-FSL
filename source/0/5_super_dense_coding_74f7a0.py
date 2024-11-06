# https://github.com/arshpreetsingh/Qiskit-cert/blob/7946e8774dfa262264c5169bd8ef14ccb5e406e0/5_super_dense_coding.py
'''
SuperDensCoding: Super-Dense Coding is communication protocol which allows user to
send two classical bit by sending only one quantum-bit.
'''
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, IBMQ, execute
from qiskit.tools.monitor import job_monitor
from read_config import get_api_key

# Connect with Backend!
IBMQ.enable_account(get_api_key())
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_qasm_simulator')

q = QuantumRegister(2, 'q')
c = ClassicalRegister(2, 'c')

##################### "00" ################################
circuit = QuantumCircuit(q,c)

### Condition for "00"
circuit.h(q[0])
circuit.cx(q[0],q[1])
circuit.cx(q[0],q[1])
circuit.h(q[0])


circuit.measure(q,c)
job = execute(circuit, backend, shots=8192)
job_monitor(job)
result = job.result().get_counts()
print("result", result)
###################### "1O" ###################################
circuit = QuantumCircuit(q,c)

# Conditon for "10"
circuit.h(q[0])
circuit.cx(q[0],q[1])
circuit.x(q[0]) # X-gate applied
circuit.cx(q[0],q[1])
circuit.h(q[0])

circuit.measure(q,c)
job = execute(circuit, backend, shots=8192)
job_monitor(job)
result = job.result().get_counts()
print("result", result)
###################### "O1" ###################################

circuit = QuantumCircuit(q,c)
# Condition for "01"
circuit.h(q[0])
circuit.cx(q[0],q[1])
circuit.z(q[0]) # Z-gate applied to q0
circuit.cx(q[0],q[1])
circuit.h(q[0])

circuit.measure(q,c)
job = execute(circuit, backend, shots=8192)
job_monitor(job)
result = job.result().get_counts()
print("result", result)
##################### "11" #######################################
circuit = QuantumCircuit(q,c)
#condition for "11"
circuit.h(q[0])
circuit.cx(q[0],q[1])
circuit.z(q[0]) # Z-gate applied
circuit.x(q[0]) # X-gate applied
circuit.cx(q[0],q[1])
circuit.h(q[0])

circuit.measure(q,c)
job = execute(circuit, backend, shots=8192)
job_monitor(job)
result = job.result().get_counts()
print("result", result)