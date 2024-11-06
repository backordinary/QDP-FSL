# https://github.com/Jawwad-Javeed/Quantum_Superposition_Distribution_Generator/blob/935f938fbb5e95bcdc3d541acd912d024d0a23e2/Python_Quantum_Superposition_Distribution_Generator/Python_Quantum_Superposition_Distribution_Generator_Polynomial.py
#Import Qiskit and Qiskit.Visualization and math for qc.ry
import qiskit
from qiskit import QuantumCircuit, assemble, Aer,execute
from qiskit.visualization import plot_histogram,plot_bloch_multivector
from math import pi

#Set Current Qiskit Backend to QASM Simulator 
#Switch if using IBM Quantum Computers
sim=Aer.get_backend('qasm_simulator')

#Intializes Quantum Circuit with 1 Qubit and 1 Classical Bit
qc=QuantumCircuit(1,1)
sim_run=1000000

#Specifies P(|1>) (Probability of |1>)
#P(|0>)=(1-P(|1>))

def Quantum_Superposition_Distribution_Polynomial(Prob_Value):
    
    #Calculates rotation required by Y axis(utilizing a Polynomial equation) in order to generate required distribution of |0> and |1>
    qc.ry(((Prob_Value/(13.402+4.378767*(Prob_Value)-0.1111414*(Prob_Value**2)))*pi),0)
    
    #Run code to check Bloch spheres of Qubits in superposition
    #qc.save_statevector()
    #qobj=assemble(qc)
    #result=sim.run(qobj).result().get_statevector()
    #plot_bloch_multivector(result)
    
    #Collapses superposition of every Qubit and assigns value to corrosponding Classical bit
    qc.measure(0,0)
    
    #Creates barrier between gates and measurements for qc.draw() and optimization level
    #qc.draw()
    
    #displays probabilities of all Qubit values
    job=execute(qc,sim,shots=sim_run)
    result=job.result()
    counts=result.get_counts()
    return plot_histogram(counts)