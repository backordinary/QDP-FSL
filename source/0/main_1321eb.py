# https://github.com/manu123416/c2q_study/blob/95aa8f95735f2b8daf65a46b2688f310eacec380/codes/main.py

### ================= ###
###  Import packages  ###
### ================= ###

import numpy as np
from PIL import Image
import time
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, Aer, execute, transpile
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from useful_methods import *
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.library.standard_gates.p import PhaseGate
import math
# Constants
method = 'c2q_m2'
img_root = '../img'
results_root = '../results'

### =========================== ###
###  Important input variables  ###
### =========================== ###

# Note: Leave this as a blank string to use random data
image = ''
# image = 'Salad_8x8.jpg'
# image = 'Salad_16x16.jpg'
# image = 'Salad_32x32.jpg'
#image = 'Salad_64x64.jpg'
#image = 'Salad_128x128.jpg'
#image = 'Salad_256x256.jpg'
# image = 'Salad_512x512.jpg'

# Number of qubits if using random data
if not image:
    num_qubits = 5

shots = 8000000
# use_complex_data = False
use_complex_data = True
### ================== ###
###  Select a backend  ###
### ================== ###

#provider = IBMQ.load_account()
#backend = provider.get_backend('ibmq_manila')
# backend = provider.get_backend('ibmq_qasm_simulator')
# backend = provider.get_backend('simulator_statevector')

backend = Aer.get_backend('qasm_simulator')
# backend = Aer.get_backend('statevector_simulator')
# backend = Aer.get_backend('aer_simulator')
# backend = Aer.get_backend('aer_simulator_statevector')

### ============================================== ###
###  Initialize the quantum circuit for the image  ###
### ============================================== ###

start_time = time.time()

if image:
    n_rows, n_cols, x_norm_not_padded, x_mag = import_image(f'{img_root}/{image}', magnitude=True)

    num_qubits = int(np.ceil(np.log2(len(x_norm_not_padded))))
    num_states = 2**num_qubits

    x_input = np.pad(x_norm_not_padded, (0, num_states-len(x_norm_not_padded)), 'constant', constant_values=0)
else:
    x_input, x_mag = random_data(num_qubits, magnitude=True, use_complex_data= use_complex_data,seed=42069)
    print(x_input)

theta, phi, r, t = input_data(x_input)
# print(theta, r, t, phi)

# create the quantum circuit for the image
qc = QuantumCircuit(num_qubits)

# Separate with barrier so it is easy to read later.
# qc.barrier()




### ================================ ###
###  Arbitrary state initialization  ###
### ================================ ###
# h_gate_qubits= []
# for i in reversed(range(num_qubits)):
#     if(i!=0):
#         h_gate_qubits.append(i)
#
# qc.h(h_gate_qubits)
# qc.barrier()

for j in reversed(range(num_qubits)):
    n_j = num_qubits - 1 - j
    i_max = 2**(n_j)
    rotations= list(theta[0:i_max,j])
    phi_rotations = list(phi[0:i_max,j])
    t_rotations= list(t[0:i_max,j])

    k= num_qubits -(j+1)
    if(k==0):
        qc.ry(rotations[0],num_qubits-1)
        qc.barrier()
        # qc.barrier()

    else:

        count = 0

        for theta_value in rotations:
            binary= bin(count)[2:]
            # print(binary)


            bitstring =(('0'* (k- len(binary))) +binary)[::-1]
            # print(bitstring)
            u= RYGate(theta= theta_value).control(num_ctrl_qubits= k, ctrl_state= bitstring)
            # print("The total qubits")
            # print(num_qubits)
            list_locations_qubits = []
            new_list = []
            # print(k)
            for i in range(k+1):
                # list_locations_qubits.append((num_qubits -1)-i)
                list_locations_qubits.append(i)
            # list_locations_qubits.reverse()
            # print(list_locations_qubits)

            new_list = [num_qubits-1- x for x in list_locations_qubits]
            qc.append(u, qargs=new_list)
            # qc.append(u, qargs= list_locations_qubits)
            count +=1

        qc.barrier()
    if j == 0:
        # if all(item == 0 for item in phi_rotations):
        count1=0
        for phi_values in phi_rotations:
            binary= bin(count1)[2:]
            bitstring = (('0'* (k- len(binary))) +binary)[::-1]
            u=RZGate(phi= phi_values).control(num_ctrl_qubits=k, ctrl_state= bitstring)

            list_locations = []
            # print(k)
            for i in range(k+1):
                list_locations.append(i)
            list_locations.reverse();
            # print(list_locations)
            qc.append(u, qargs= list_locations)
            list_locations.clear()

            count1 +=1
        qc.barrier()

    if j == 0:
        # if all(item == 0 for item in t_rotations):
        count2=0
        for t_values in t_rotations:
            binary = bin(count2)[2:]
            bitstring = (('0' * (k - len(binary))) + binary)[::-1]
            u = PhaseGate(theta=t_values).control(num_ctrl_qubits=k, ctrl_state=bitstring)
            list_qubits_loc = []
            for i in range(k + 1):
                list_qubits_loc.append(i)
            list_qubits_loc.reverse()
            qc.append(u, qargs=list_qubits_loc)
            list_qubits_loc.clear()
            count2 += 1
        qc.barrier()



small_font = {
     "fontsize": 10,
     "subfontsize": 6,
}
# Large images exceed the max figure size of the notebook
if not image:
    qc.draw(output='mpl',style= small_font)

plt.show()

### ================================== ###
###  Measurement and circuit analysis  ###
### ================================== ###


qc.measure_all()

qc.barrier()

t_setup = time.time() - start_time # seconds

counts, t_exec = run_circuit(qc, shots, backend)

print(f"Setup time: {t_setup} sec\nExecution Time: {t_exec} sec\nTotal Time: {t_setup + t_exec} sec")

x_expected = x_input * x_mag
x_output = get_data(counts, shots, magnitude=x_mag, num_qubits=num_qubits)
x_err = np.abs(x_output-np.abs(x_expected)) / np.abs(x_expected)

for i, (x_in, x_out, err) in enumerate(zip(x_expected, x_output, x_err)):
    print(f"{i}: Expected {x_in:.3f}, Got {x_out:.3f}, Err of {100*err:.3f}%")

## Image Reconstruction

### ====================== ###
###  Image reconstruction  ###
### ====================== ###

if image:
    img_data = x_output[0:len(x_norm_not_padded)] # Remove padded zeroes
    img_data = np.ceil(img_data) # Eliminates decimal points
    img_data = np.uint8(img_data) # Image package needs uint8
    img_data = np.reshape(img_data, ((n_rows, n_cols, 3)))
    img = Image.fromarray(img_data, 'RGB')

    img.save(f'{results_root}/{method}/{image}')


### ============================== ###
###  Write results to output file  ###
### ============================== ###

with open(f'{results_root}/{method}/{image}-results.txt', 'w') as f:
    f.write(str(qc.num_qubits))
    f.write('\n')

    f.write(f"{t_setup}\n{t_exec}\n{t_setup + t_exec}")
    f.write('\n')

    fid = get_fidelity(np.abs(x_input), x_output / x_mag)
    f.write(str(fid))
    print(f'Image fidelity: {100*fid:.3f}%')