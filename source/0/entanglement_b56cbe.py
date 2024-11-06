# https://github.com/qismib/QuITE/blob/41ab15d03e2a3544533e889bb11b16bf148c8e28/programs/entanglement.py
#primo programma: entanglement tra due qubit 
#da modificare in una funzione per il quantum teleportation

from qiskit import *
from qiskit.visualization import *
from qiskit.tools.monitor import job_monitor
import matplotlib
import matplotlib.pyplot as plt

#loading my IBMQ account
provider = IBMQ.load_account()  

#configuaration
qr = QuantumRegister(2)
cr = ClassicalRegister(2)
circuit = QuantumCircuit(qr, cr)



#now we create entanglement 
circuit.h(qr[0])
circuit.cx(qr[0],qr[1])

#now we measure the qubits
#and we store the measurements in classical bits
circuit.measure(qr,cr)

circuit.draw(output = 'mpl') #let's see

#local simulation 
simulator = Aer.get_backend('qasm_simulator')
simulation = execute(circuit,backend = simulator)
result = simulation.result()

plot_histogram(result.get_counts(circuit))

print("simulazione conclusa")

plt.show()

#now on a real quantum computer
realQc = provider.get_backend('ibmq_5_yorktown')
job = execute(circuit, backend = realQc)

job_monitor(job)
reasults = job.result()

plot_histogram(reasults.get_counts(circuit))
print("yorktown ha concluso")

plt.show()

















