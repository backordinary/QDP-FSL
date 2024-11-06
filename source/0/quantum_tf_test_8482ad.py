# https://github.com/OrdinaryHacker101/QiskitTensorflowExperiments/blob/1db62ed964c552e8f9dbc0b2687e6100fc7ab4d6/quantum_tf_test.py
##import tensorflow as tf
import numpy as np #numpy
import qiskit as q #qiskit
from qiskit import *
import time #to see how long it takes for the program to run

print("Quantum computer is starting")
start_time = time.time() #start time for quantum program

#ibm account code for quantum computer
q.IBMQ.save_account("4842dc61d90a83c6c305c6ef6fc75c2aa65af5b26bd202989032ae14b7df56c2d9cc0e14ac1a3847ecb539b214cf9f08b894dfeb1f72ac291a2d57c19729d1b3")
IBMQ.load_account() #login

provider = q.IBMQ.get_provider("ibm-q") #get available quantum computers

backend = provider.get_backend("ibmq_santiago") #computer from santiago, since it has alot of qubits

qc = q.QuantumCircuit(2, 2) #quantum circuit
qc.x(0) #flip all the 0's to 1's and 1's to 0's
qc.measure([0,1], [0,1]) #turn the qubit results to classical bits

job = q.execute(qc, backend=backend, shots = 50) #execute the circuit
result = job.result() #getting results
counts = result.get_counts(qc) #getting the counts for qiskit

end_time = time.time() #finish time

print("The quantum computer took: ", end_time-start_time)

print(np.array(counts)) #displays the whole result
print(np.array(counts["01"])) #prints the result the 2nd time, but this time only the "01" state is shown

#in future testing tensorflow is yet to be used

##layer = tf.keras.layers.Dense(100)
##layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
##print(layer(tf.zeros([10,5])))
