# https://github.com/stoicswe/CSCI395A-QuantumComputing/blob/7bd82873d4477ca0b4daab96964ebc44b9036671/amplitudePhase/phaseGates.py
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

import os
sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/nathanbunch/qiskit-sdk-py'))
    
# useful additional packages 
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

# importing the QISKit
from qiskit import QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

backend = 'local_qasm_simulator' # the backend to run on
shots = 1024    # the number of shots in the experiment 

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url


# Creating registers
qr = Q_program.create_quantum_register("qr", 1)
cr = Q_program.create_classical_register("cr", 1)
circuits = []

phase_vector = range(0,100)
for phase_index in phase_vector:
    phase_shift = phase_index-50
    phase = 2*np.pi*phase_shift/50
    circuit_name = "phase_gate_%d"%phase_index
    qc_phase_gate = Q_program.create_circuit(circuit_name, [qr], [cr])
    qc_phase_gate.h(qr)
    qc_phase_gate.u1(phase, qr)
    qc_phase_gate.h(qr)
    qc_phase_gate.measure(qr[0], cr[0])
    circuits.append(circuit_name)

result = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)

probz = []
phase_value = []
for phase_index in phase_vector:
    phase_shift = phase_index - 50
    phase_value.append(2*phase_shift/50)
    if '0' in result.get_counts(circuits[phase_index]):
        probz.append(2*result.get_counts(circuits[phase_index]).get('0')/shots-1)
    else:
        probz.append(-1)

plt.plot(phase_value, probz, 'b',0.25,1/np.sqrt(2),'ro',0.5,0,'ko',1,-1,'go',-0.25,1/np.sqrt(2),'rx',-0.5,0,'kx',-1,-1,'gx')
plt.xlabel('Phase value (Pi)')
plt.ylabel('Eigenvalue of X')

plt.show()

backend = 'local_qasm_simulator' # the backend to run on
shots = 1024    # the number of shots in the experiment 

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url


# Creating registers
qr = Q_program.create_quantum_register("qr", 1)
cr = Q_program.create_classical_register("cr", 1)
circuits = []

phase_vector = range(0,100)
for phase_index in phase_vector:
    phase_shift = phase_index-50
    phase = 2*np.pi*phase_shift/50
    circuit_name = "phase_gate_%d"%phase_index
    qc_phase_gate = Q_program.create_circuit(circuit_name, [qr], [cr])
    qc_phase_gate.u3(phase,0,np.pi, qr)
    qc_phase_gate.measure(qr[0], cr[0])
    circuits.append(circuit_name)

result = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)

probz = []
phase_value = []
for phase_index in phase_vector:
    phase_shift = phase_index - 50
    phase_value.append(2*phase_shift/50)
    if '0' in result.get_counts(circuits[phase_index]):
        probz.append(2*result.get_counts(circuits[phase_index]).get('0')/shots-1)
    else:
        probz.append(-1)

plt.plot(phase_value, probz, 'b',0.5,0,'ko',1,-1,'go',-0.5,0,'kx',-1,-1,'gx')
plt.xlabel('Phase value (Pi)')
plt.ylabel('Eigenvalue of Z')

plt.show()