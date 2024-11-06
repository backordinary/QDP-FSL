# https://github.com/theheavygluon/QASM-Interpreter/blob/4676499bd0a0ee28d68a301d9f5f83df616ae071/interpreter.py
from qiskit import *
import numpy as np




#Supplementary function that gets the counts and plots them 
def plot_counts(qc,shots=3000): 
    assert qc.num_clbits > 0, "There are no classical bits to measure."
    sim = Aer.get_backend('qasm_simulator')
    counts = execute(qc,sim,shots=shots).result().get_counts()
    return plot_histogram(counts)

#Supplementary function that gets the circuit unitary
def get_unitary(circ):
    sim = Aer.get_backend('unitary_simulator')
    U = execute(circ,sim).result().get_unitary()
    return U

#Converts QASM file to QASM string
def file_to_str(filename):
    return open(filename,'r').read()


#Function that converts the QASM format into a list that would make interpreting more convenient

def qasm_list(string):
    string = string.replace("\n", "")
    return string.split(";")

#Function that formats and filters the instructions
def instruction_gen(string):
    
    inst_set = string.split(" ")
    #If parameter gate, the formatting is slightly more complex given the parameter is written inside the gate
    if "rx" in inst_set[0] or "ry" in inst_set[0] or "rz" in inst_set[0]:
        gate = inst_set[0][0:2]
        angle = inst_set[0].replace(gate+"(","")
        angle = angle.replace(")","")
        angle = angle.replace("pi","3.14159265")
        target = inst_set[1].replace("q[","")
        target = [float(eval(angle))]+[int(i) for i in target.replace("]","").split(",")] 
    #Measuring also has a slightly different format
    elif "measure" in inst_set[0]:
        gate = "measure"
        target = string.replace('measure q[',"")
        target = target.replace("] -> c["," ") 
        target = [[int(i)] for i in target.replace("]","").split(" ")]
    #Rest of the gates
    else:
        gate = inst_set[0]
        target = inst_set[1].replace("q[","")
        target = [int(i) for i in target.replace("]","").split(",")]
    return [gate,target]

#Function that creates circuits
def make_circuit(string):
    words = string
    n_qb = 0
    n_cb = 0  
    qb = [w for w in words if w.startswith('qreg')][0]
    qb = qb.replace("qreg q[","")
    n_qb = int(qb.replace("]","")) #Identifies the number of classical bits
    if [w for w in words if w.startswith('creg')] != []:
        cb = [w for w in words if w.startswith('creg')][0]
        cb = cb.replace("creg c[","") #Identifies whether there are classical bits and finds them
        n_cb = int(cb.replace("]","")) 
    if n_cb == 0:
        qc = QuantumCircuit(n_qb) #Builds quantum circuit with only qubits
    if n_cb != 0:
        qc = QuantumCircuit(n_qb, n_cb) #Builds quantum circuit with both qubits and classical bits
    return qc

#The main show: Implements all the functions defined before and creates a qiskit QuantumCircuit object, equivalent to the input qasm string

def interpreter(qasm_in):    
    #Adds instructions, line by line
    def add_instructions(qasm_str):
        allowed_gates = ('x','y','z','rx','ry','rz','h','s','sdg','t','tdg','cx','ccx','swap','cswap','measure') # Allowed gates in problem statement
        for i in range(len(qasm_str)):
            if qasm_str[i].startswith(allowed_gates):
                gate,target = instruction_gen(qasm_str[i])
                getattr(qc, gate)(*target) #Evaluates gate operation (which is stored as a string)
                
    if ".qasm" in qasm_in: 
        qasm_str = file_to_str(qasm_in) #Converts file to string 
    else:
        qasm_str = qasm_in 

    qasm = qasm_list(qasm_str) #Turns string to list 
    qc = make_circuit(qasm) #Creates the circuit
    add_instructions(qasm) #Adds instructions 
    return qc #Outputs the circuit 

#Gets dagger of circuit 

def get_dg(qc):
    
    if type(qc) == str: #If the input is the qasm string
        qc = interpreter(qc) #Converts to quantum circuit
        u_dg = get_unitary(qc).conjugate().transpose() #Gets u_dg
        qc_dg = QuantumCircuit(qc.num_qubits) 
        qc_dg.unitary(u_dg,[i for i in range(qc.num_qubits)],"qc_dg") #Appends u_dg as a gate to new circuit. Could be appended to other circuits using normal qiskit code
    
    if type(qc) == qiskit.circuit.quantumcircuit.QuantumCircuit: #If the input is directly a quantum circuit
        u_dg = get_unitary(qc).conjugate().transpose() 
        qc_dg = QuantumCircuit(qc.num_qubits)
        qc_dg.unitary(u_dg,[i for i in range(qc.num_qubits)],"qc_dg")
    return qc_dg