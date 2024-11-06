# https://github.com/CodieKev/Variational_Quantum_Classifier_CKT/blob/a10dab69a2434ed6c9bd701fec6c13311b6d7b9f/Custom_4_Feature_Mapping_Custom_Classifier_v1/variational_circuit.py
# the write_and_run function writes the content in this cell into the file "variational_circuit.py"

### WRITE YOUR CODE BETWEEN THESE LINES - START
    
# import libraries that are used in the function below.
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import  RealAmplitudes, EfficientSU2
    
### WRITE YOUR CODE BETWEEN THESE LINES - END
import math

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

def variational_circuit():
    # BUILD VARIATIONAL CIRCUIT HERE - START
    
    # import required qiskit libraries if additional libraries are required
    
    # build the variational circuit
    #var_circuit = EfficientSU2(num_qubits=3, su2_gates= ['rx', 'ry'], entanglement='circular', reps=3)
    #var_circuit = EfficientSU2(num_qubits=4, su2_gates= ['rx', 'ry'], entanglement='circular', reps=3)
    
    # BUILD VARIATIONAL CIRCUIT HERE - END
    
    # return the variational circuit which is either a VaritionalForm or QuantumCircuit object
    from qiskit.circuit import QuantumCircuit, ParameterVector

    num_qubits = 4            
    reps = 2              # number of times you'd want to repeat the circuit
    reps_2 = 2

    x = ParameterVector('x', length=reps*(5*num_qubits-1))  # creating a list of Parameters
    qc = QuantumCircuit(num_qubits)

    # defining our parametric form
    for k in range(reps):
            for i in range(num_qubits):
                qc.rx(x[2*i+k*(5*num_qubits-1)],i)
                qc.ry(x[2*i+1+k*(5*num_qubits-1)],i)
            for i in range(num_qubits-1):
                qc.cx(i,i+1)
            for i in range(num_qubits-1):
                qc.rz(2.356194490192345, i)
                qc.rx(1.5707963267948966, i)
                qc.rz(-2.356194490192345, i+1)
                qc.rx(1.5707963267948966, i+1)
                qc.cz(i, i+1)
                qc.rz(-1.5707963267948966, i)
                qc.rx(1.5707963267948966, i)
                qc.rz(x[i+2*(num_qubits)+k*(5*num_qubits-1)], i)
                qc.rx(-1.5707963267948966, i)
                qc.rz(1.5707963267948966, i+1)
                qc.rx(1.5707963267948966, i+1)
                qc.rz(x[i+2*(num_qubits)+k*(5*num_qubits-1)], i+1)
                qc.rx(-1.5707963267948966, i+1)
                qc.cz(i, i+1)
                qc.rz(-1.5707963267948966, i)
                qc.rx(1.5707963267948966, i)
                qc.rz(0.7853981633974483, i)
                qc.rz(-1.5707963267948966, i+1)
                qc.rx(-1.5707963267948966, i+1)
                qc.rz(2.356194490192345, i+1)
            for i in range(num_qubits):
                qc.rx(x[2*i+3*num_qubits-1+k*(5*num_qubits-1)],i)
                qc.ry(x[2*i+1+3*num_qubits-1+k*(5*num_qubits-1)],i)
        
            
    
                
            
    #custom_circ.draw()
    return qc
