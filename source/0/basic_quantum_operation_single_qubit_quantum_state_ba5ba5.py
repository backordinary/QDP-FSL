# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutoriale-QISKit/01-getting_started/basic_quantum_operation_single_qubit_quantum_state.py
import sys
import os

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

    # useful additional packages
    import matplotlib.pyplot as plt 
    import numpy as np 
    from math import pi

    from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, QISKitError
    from qiskit import available_backends, execute, register, get_backend
    from qiskit.tools.visualisation import circuit_drawer

    # Define the Classical and Quantum Registers
    c = ClassicalRegister(1)
    q = QuantumRegister(1)

    # Define Quantum Circuit
    qc = QuantumCircuit(q, c)
    qc.u3(pi/2,pi/2,pi/2,q)
    circuit_drawer(qc)