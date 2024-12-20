# https://github.com/adlantz/QiskitSG/blob/ad81cc441d5ffaee38a704ad92cdb193e36c613d/SGFrustration.py
#initialization
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import os



# importing Qiskit
from qiskit import IBMQ, Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.extensions.simulator import snapshot
from qiskit.providers.ibmq import least_busy
from qiskit.quantum_info import Statevector

# import basic plot tools
from qiskit.visualization import plot_histogram



def main():
    #Get number of spins
    N = int(input("Number of spins/qubits: "))
    bond_list = bond_list_maker(N)
    bqb = len(bond_list)


    #Build requisite quantum and classical registers
    spin_qubits = QuantumRegister(N, name='v')
    bond_qubits = QuantumRegister(bqb, name='c')
    output_qubit = QuantumRegister(1, name='out')
    cbits = ClassicalRegister(bqb, name='cbits')
    qc = QuantumCircuit(spin_qubits, bond_qubits, output_qubit, cbits)


    # Initialise 'out0' in state |->
    qc.initialize([1, -1]/np.sqrt(2), output_qubit)

    # Initialise qubits in state |s>
    qc.h(spin_qubits)
    qc.h(bond_qubits)
    # qc.h(t_qubits)
    qc.barrier()  # for visual separation


    # Determine how many times to loop Uw/Us operators
    loopdict = {3:2, 4:6, 5:24}
    if N in loopdict:
    	loopnumber = loopdict[N]
    else:
    	loopnumber = 25
    for i in range(loopnumber):
        SG_oracle(qc, bond_list, spin_qubits, bond_qubits, cbits, output_qubit)
        qc.barrier()  # for visual separation
        # Apply our diffuser
        qc.append(diffuser(bqb), bond_qubits)


    # Measure the variable qubits
    # qc.measure(bond_qubits, cbits)



    #Print/Save circuit interface
    asking=True
    while(asking):
        prntqc = input("Print Circuit? (y/n)")
        if prntqc == 'y' or prntqc == 'yes':
            print(qc.draw())
            asking = False
        elif prntqc == 'n' or prntqc == 'no':
            asking = False
        else:
            print("invalid input")
    
    asking=True
    while(asking):
        sveqc = input("Save circuit as png? (y/n)")
        if sveqc == 'y' or sveqc == 'yes':
            print("saving to " + str(os.getcwd()))
            qc.draw(output='mpl',filename="N" + str(N) + "_frust_qc.png")
            asking = False
        elif sveqc == 'n' or sveqc == 'no':
            asking = False
        else:
            print("invalid input")




    print("Simulating quantum circuit...")

    # Send circuit to backend
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    result = job.result()
    outputstate = result.get_statevector(qc)
    probstate = np.multiply(outputstate,np.conj(outputstate)).real


    #get most probable states out of quantum state after algorithm has run
    probampdict = {}
    bs = '{0:0' + str(bqb) + 'b}'
    for i in range(2**bqb):
        probampdict[bs.format(i)] = 0

    i = 0
    fbs = '{0:0' + str(bqb + N + 1) + 'b}'
    for ops in probstate:
        probampdict[fbs.format(i)[1:(bqb+1)]] += ops
        i+=1



    solpa = probampdict[bs.format((2**(bqb) - 1))]


    i=0
    j=1
    bcarray=[]
    for key in probampdict:
        if probampdict[key] > 0.9*solpa:
            print(str(j) +".",bs.format(i)[::-1],probampdict[key])
            bcarray.append(bs.format(i))
            j+=1
        i+=1
    


    #Give the option to visualize bond configurations using SGViz.py 
    viz = True
    invld = False
    while(viz):
        index = input("Visualize circuit (input number of configuration from list above or q to quit): ")
        try:
            int(index)
            invld= False
        except:
            invld = True
        if index == 'q' or index == 'quit':
            break
        elif invld:
            print("Invalid input")
        elif int(index) <= len(bcarray):
            os.system("python3 SGViz.py -loadBC -BC " + str(bcarray[int(index)-1]) + " -N " + str(N))
        else:
            print("not a valid index")
    

    print("Goodbye")

def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc

def bond_list_maker(N):
    return np.array([np.array(list(c)) for c in it.combinations([i for i in range(N)],2)])


def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "$U_s$"
    return U_s


def XOR(qc, a, b, output):
    qc.cx(a, output)
    qc.cx(b, output)
    qc.barrier()

def XNOR(qc, a, b, output):
    qc.cx(a, output)
    qc.cx(b, output)
    qc.x(output)
    qc.barrier()

def SG_oracle(qc, bond_list, spin_qubits, bond_qubits, cbits, output_qubit):
    # Compute clauses
    i = 0
    for clause in bond_list:
        XOR(qc, clause[0], clause[1], bond_qubits[i])
        i += 1

    # Flip 'output' bit if all clauses are satisfied
    qc.mct(bond_qubits, output_qubit)

    # Uncompute clauses to reset clause-checking bits to 0
    i = 0
    for clause in bond_list:
        XOR(qc, clause[0], clause[1], bond_qubits[i])
        i += 1



if __name__ == '__main__':
    main()