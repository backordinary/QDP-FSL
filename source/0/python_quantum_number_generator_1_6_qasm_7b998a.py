# https://github.com/J2304789/Quantum-Random-Number-Generator/blob/dd751ffb653b88fdf4610ea3a5ba54e4efb5caa9/Python_Quantum_Random_Number_Generator/Python_Quantum_Number_Generator_1-6_QASM.py
#Import Qiskit and Qiskit.Visualization
import qiskit
from qiskit import QuantumCircuit, assemble, Aer,execute
from qiskit.visualization import plot_histogram,plot_bloch_multivector
#Set Current Qiskit Backend to QASM Simulator 
#Switch if using IBM Quantum Computers
sim=Aer.get_backend('qasm_simulator')

#Intializes 3 Qubits and 3 Classical Bits
qc=QuantumCircuit(3,3)

#Amounts of times Simulation is run
sim_run=1
qc.h(0)
qc.h(1)
qc.h(2)
#Creates barrier between gates and measurements for qc.draw() and optimization level
qc.barrier()
qc.measure(0,0)
qc.measure(1,1)
qc.measure(2,2)
#Function to convert Qubit Binary to Base 10 and displays randomly generated number
def QASM_Generate():
    #memory=True to access indivual simulation qubit measurement values
    job=execute(qc,sim,shots=sim_run,memory=True)
    result=job.result()
    counts=result.get_counts()
    memory=result.get_memory()
    #Converts Qubit Binary to Base 10
    int_value=int(memory[0],2)
    
    #Check int_value throughout iterations
    #print(int_value)
    
    if int_value==7 or int_value==0:
        QASM_Generate()
    else:
        return int_value
        

#QASM_Generate()