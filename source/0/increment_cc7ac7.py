# https://github.com/reverseredoc/SoC-2022/blob/16a5bc0412142b92407eea0ab5169948003a2d09/week2/increment.py
#increment

# imports
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
from qiskit.extensions import Initialize
from qiskit.ignis.verification import marginal_counts
from qiskit.quantum_info import random_statevector

def initialiser(string,add_circuit):
       
    if string[0]=='1':
        add_circuit.x(2-0)
       
    if string[1]=='1':
        add_circuit.x(2-1)
       
    if string[2]=='1':
        add_circuit.x(2-2)

def zero_control(a,b,c,add_circuit):
    add_circuit.x(b)
    add_circuit.ccx(a,b,c)
    add_circuit.x(b)
    
def addition_circuit(b):
    
    b_str = format(b, '0'+str(3)+'b')
    add_circuit=QuantumCircuit(3,3)
    initialiser(b_str,add_circuit)
    if b!=0:
        add_circuit.barrier()
    add_circuit.cx(0,1)
    zero_control(0,1,2,add_circuit)
    add_circuit.x(0)
    return add_circuit
    
b =np.random.randint(0,2**3)
print("Input integer:",b)

circuit=addition_circuit(b)

circuit.measure_all()

display(circuit.draw())

svsim = Aer.get_backend('aer_simulator')

circuit.save_statevector()
qobj = assemble(circuit)
final_state = svsim.run(qobj).result().get_statevector()

from qiskit.visualization import array_to_latex
display(array_to_latex(final_state, prefix="\\text{Statevector} = "))
if final_state[0]!=0:
    print("Output integer:",0)
    break
if final_state[1]!=0:
    print("Output integer:",1)
    break
if final_state[2]!=0:
    print("Output integer:",2)
    break
if final_state[3]!=0:
    print("Output integer:",3)
    break
if final_state[4]!=0:
    print("Output integer:",4)
    break
if final_state[5]!=0:
    print("Output integer:",5)
    break
if final_state[6]!=0:
    print("Output integer:",6)
    break
if final_state[7]!=0:
    print("Output integer:",7)
    break
