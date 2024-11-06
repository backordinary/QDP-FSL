# https://github.com/39xdgy/selflearn_qiskit/blob/874d6aceb8487ec1cd66f5da7c516d54ba541bab/intro_qiskit.py
#import qiskit

#qiskit.__qiskit_version__



from qiskit import IBMQ
from qiskit import *
import matplotlib.pyplot as plt


qr = QuantumRegister(2)

cr = ClassicalRegister(2)

circuit = QuantumCircuit(qr, cr)
#%matplotlib inline


circuit.h(qr[0])  #hadamard gate

#print(circuit.draw(filename = "output.jpg", output = 'mpl'))


circuit.cx(qr[0], qr[1])  #cnot, equal to if statment

#print(circuit.draw(filename = "output.jpg", output = 'mpl'))


circuit.measure(qr, cr)

#circuit.draw( initial_state=True, output = 'mpl')

#plt.show()

simulator = Aer.get_backend('qasm_simulator')

result = execute(circuit, backend = simulator).result()


from qiskit.tools.visualization import plot_histogram

plot_histogram(result.get_counts(circuit))



#IBMQ.save_account('token', overwrite = True)
'''
provider = IBMQ.load_account()

backend = provider.backends.ibmq_vigo

qobj = assemble(transpile(circuit, backend = backend), backend = backend)

job = backend.run(qobj)
'''
from qiskit.tools.monitor import job_monitor, backend_overview
IBMQ.load_account()

provider = IBMQ.get_provider('ibm-q')

backend_overview()

backend_name = input("Which backend do you want to use: ")
qcomp = provider.get_backend(backend_name)

job = execute(circuit, backend = qcomp)


job_monitor(job)

result = job.result()

plot_histogram(result.get_counts(circuit))

plt.show()
