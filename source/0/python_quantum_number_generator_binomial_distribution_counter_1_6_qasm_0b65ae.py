# https://github.com/J2304789/Quantum-Random-Number-Generator/blob/dd751ffb653b88fdf4610ea3a5ba54e4efb5caa9/Python_Quantum_Random_Number_Generator/Python_Quantum_Number_Generator_Binomial_Distribution_Counter_1-6_QASM.py
#Import Qiskit
import qiskit
from qiskit import QuantumCircuit, assemble, Aer,execute

#Set Current Qiskit Backend to QASM Simulator 
#Switch if using IBM Quantum Computers
sim=Aer.get_backend('qasm_simulator')

#Intializes 1 Qubit and 1 Classical Bit
qc=QuantumCircuit(1,1)

#Amounts of times Simulation is run
sim_run=6

#Initial Value of Binomial Distribution count
Start_value=0
qc.h(0)
#Creates barrier between gates and measurements for qc.draw() and optimization level
qc.barrier()
qc.measure(0,0)
#Function to convert Qubit Binary to Base 10 and displays randomly generated number
def Binomial_Distribution_Counter_Generate():
    #memory=True to access indivual simulation qubit measurement values
    job=execute(qc,sim,shots=sim_run,memory=True)
    result=job.result()
    counts=result.get_counts()
    memory=result.get_memory()

    for x in range(0,sim_run):
        global Start_Value
        int_value=int(memory[x],2)
        #Adds 1 to Start_Value if |1> else adds 0
        Start_Value=Start_Value+int_value
        
    return Start_Value

#Binomial_Distribution_Counter_Generate()