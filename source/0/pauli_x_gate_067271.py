# https://github.com/keshavbharadwaj/Quantum-circuits-in-python/blob/0c620f78180e6515554855ae333de496f9670530/Basic%20programs/pauli_X_gate.py
import qiskit as q
from qiskit import Aer,execute
from qiskit import IBMQ
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import matplotlib
f=open('ibmtoken.txt')
token=f.read()
IBMQ.load_account()
qr=q.QuantumRegister(2)
cr=q.ClassicalRegister(2)
circuit=q.QuantumCircuit(qr,cr)
circuit.x(qr[0])
circuit.cx(qr[0],qr[1])
##circuit.measure([qr[0],qr[1]],[cr[0],cr[1]])#measureing and storing result in classical bits
##simulator=Aer.get_backend('qasm_simulator')#using the qasm simulator 
##Result=execute(circuit,backend=simulator).result()
##plot_histogram(Result.get_counts(circuit)).show()
##print(circuit)
##circuit.draw(output = 'mpl',filename='ckt.png')
provider=IBMQ.get_provider('ibm-q')#using actual quantum computer
qcomp=provider.get_backend('ibmq_16_melbourne')#using the ibmq_16_melbourne
job=execute(circuit,backend=qcomp)
job_monitor(job)#monitoring for the completion of our job
result=job.result()
plot_histogram(result.get_counts(circuit)).show()#plotting the results
