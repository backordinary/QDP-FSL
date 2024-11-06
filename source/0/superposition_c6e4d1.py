# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutorial_programming_IBM/realQC/1/superposition.py
import sys
import os
# Checking the version of PYTHO; we only support > 3.5
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

# import qiskit from qiskit-sdk-py folder 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'qiskit-sdk-py'))
from qiskit import QuantumProgram

# Creating Programs create your 1st QuantumProgram object instance
Q_program = QuantumProgram()

# Set your API Token
# You cant get it from your IBM QX account
# looking for "Personal Access Token" section
QX_TOKEN = "da2a4002660558a35103a600bcbda7fe438cea629a6be98969ea5e367c091b6815e624bd86b6207121bd97fef79c22033318a4402eeafcbd04b021fd80f5a195"
QX_URL = "https://quantumexperience.ng.bluemix.net/api"

# Set up the API and execute the program
# You need the API Token and the QX URL
Q_program.set_api(QX_TOKEN, QX_URL)

# Creating Registers 
# create your 1st Quantum Register called "qr" with 2 qubits
qr = Q_program.create_quantum_register("qr", 2)
# create your 1st Classical Register called "cr" with 2 bits
cr = Q_program.create_classical_register("cr", 2)

# Createing Circuits
# Create your 1st Quantum Circuit called "qc" composed from your Quantum Register "qr" and your Classical Register "cr"
qc = Q_program.create_circuit("superposition", [qr], [cr])

# add the H gate in the Qubit 0, we put this Qubit in superposition
qc.h(qr[0])

# add measure to see the state
qc.measure(qr, cr)

# Compiled and execute in the local_qasm_simulator

result = Q_program.execute(["superposition"], backend = 'ibmqx2', shots=1024)

# Show the results
print(result)
print(result.get_data("superposition"))