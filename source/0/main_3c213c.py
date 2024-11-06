# https://github.com/BryceFuller/quantum-mobile-backend/blob/5211f07813e581a00d6cee9e1b02dea55d7d7f50/main.py
from flask import Flask, jsonify, request, make_response
import sys
import qiskit
#import qiskit.tools.visualization
# if sys.version_info < (3,5):
#     raise Exception('Please use Python version 3.5 or greater.')

# Import basic plotting tools
#from qiskit.tools.visualization import plot_histogram
# Importing QISKit
from qiskit import QuantumCircuit, QuantumProgram
import Qconfig
import numpy as np
app = Flask(__name__)

@app.route("/", methods=['POST'])
def hello():
	res_string = "No data returned"
	to_run = request.get_json(force=True)	#TEST
	
	# Quantum program setup 
	Q_program = QuantumProgram()
	Q_program.set_api(Qconfig.APItoken, Qconfig.config['url']) # set the APIToken and API url
	# Creating registers
	q = Q_program.create_quantum_register('q', 3)
	c0 = Q_program.create_classical_register('c0', 1)
	c1 = Q_program.create_classical_register('c1', 1)
	c2 = Q_program.create_classical_register('c2', 1)

	# Quantum circuit to make the shared entangled state 
	teleport = Q_program.create_circuit('teleport', [q], [c0,c1,c2])
	teleport.h(q[1])
	teleport.cx(q[1], q[2])

	teleport.ry(np.pi/4,q[0])

	teleport.cx(q[0], q[1])
	teleport.h(q[0])
	teleport.barrier()  

	teleport.measure(q[0], c0[0])
	teleport.measure(q[1], c1[0])

	circuits = ['teleport']
	print(Q_program.get_qasms(circuits)[0])

	# backend = 'ibmqx2' # the backend to run on
	backend = 'local_qasm_simulator' 
	shots = 1024 # the number of shots in the experiment 

	result = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240)

	print(result.get_counts('teleport'))
	data = result.get_counts('teleport')
	return str(data);


	'''print(data);
	alice = {}
	alice['00'] = data['0 0 0'] + data['1 0 0']
	alice['10'] = data['0 1 0'] + data['1 1 0']
	alice['01'] = data['0 0 1'] + data['1 0 1']
	alice['11'] = data['0 1 1'] + data['1 1 1']
	plot_histogram(alice)

	bob = {}
	bob['0'] = data['0 0 0'] + data['0 1 0'] +  data['0 0 1'] + data['0 1 1']
	bob['1'] = data['1 0 0'] + data['1 1 0'] +  data['1 0 1'] + data['1 1 1']
	plot_histogram(bob)
	'''

	#for y in range(0, len(list)):
	 #   print(list[0])

	return "Test"
	# Do magic
	#return res_string
