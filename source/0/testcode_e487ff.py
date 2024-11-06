# https://github.com/ayush0624/QIGA2-Tuned/blob/924928d6d4532d9456cb883903e25b4e05ea70a4/Code/QIGA-2/TestCode.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, BasicAer, IBMQ
from qiskit.providers.aer import noise

# Choose a real device to simulate
APItoken = '5b2f71479eae0159258df0ece626df4f137a6fa7126058500c086b17aa23333b244c003475c8b7f1c2c162c52fc81f2d272d300881e872cf1ba28a3060afe090'
url = 'https://quantumexperience.ng.bluemix.net/api'

IBMQ.enable_account(APItoken, url=url)
IBMQ.load_accounts()
realBackend = IBMQ.backends(name='ibmqx2')[0]
device = IBMQ.get_backend(realBackend)
coupling_map = device.configuration().coupling_map



# Generate a quantum circuit
q = QuantumRegister(2)
c = ClassicalRegister(2)
qc = QuantumCircuit(q, c)

qc.h(q[0])
qc.cx(q[0], q[1])
qc.measure(q, c)

# Perform noisy simulation
backend = BasicAer.get_backend('qasm_simulator_py')
job_sim = execute(qc, realBackend,
                  coupling_map=coupling_map)
sim_result = job_sim.result()

print(sim_result.get_counts(qc))