# https://github.com/gowtham07/demo_quantum_docker_ibm/blob/11d2f52b5fae75ce1b5d12670fa68cfe87174668/circuit.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, transpile, Aer, assemble
from qiskit.tools.monitor import job_monitor

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

qr = QuantumRegister(5)
cr = ClassicalRegister(5)

circuit = QuantumCircuit(qr, cr)
circuit.x(qr[0])
circuit.x(qr[1])
circuit.ccx(qr[0], qr[1], qr[2])
circuit.cx(qr[0], qr[1])
circuit.measure(qr, cr)


IBMQ.save_account("283d8c0a0678ffee8e20db25a12e04bde19130d3aacca46ff1d02d5473fee63d97b24445cb4f3977ef351b723e5fd46b342d97ee16b09715473fc11ad449c2e3", overwrite=True) 
provider = IBMQ.load_account()

# Get backend for experiment
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_quito')

# prepare the circuit for the backend
mapped_circuit = transpile(circuit, backend=backend)
qobj = assemble(mapped_circuit, backend=backend, shots=1024)

# execute the circuit
job = backend.run(qobj)

job.status()

while str(job.status())== 'JobStatus.QUEUED':
    job.status()


    

job = provider.get_backend('ibmq_quito').retrieve_job(job.job_id())

result = job.result()
counts = result.get_counts()
print(counts)


