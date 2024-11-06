# https://github.com/pricoptudor/Quantum/blob/acf20bcc97cc22e3c4584f876954c434092b3360/Homework-2/ex1.py
from qiskit import *
from qiskit.visualization import *
from qiskit.circuit.library import *
from qiskit.circuit import *
import numpy as np
import cmath

epsilon=np.finfo(complex).eps * 100 #pentru epsilon normal matricile au elemente diferite, chiar daca ar trebui sa fie egale
raspuns=1

#portile sunt egale: control-0 control-1 control-0  inseamna XIX + control-1 control-1 control-1 + XIX
#                    U^3 = U^2 * U = I * U = U (pentru X,I,Z,H)
#                    S^2 != I  ;   S^3 = S^-1

def egalitate_matrici(m1,m2):
    if(len(m1)!=len(m2)):
        return ('Dimensiuni incompatibile')
    if(len(m1[0])!=len(m2[0])):
        return ('Dimensiuni incompatibile')
    for i in range(len(m1)):
        for j in range(len(m2)):
            if(abs(m1[i][j]-m2[i][j])>epsilon):
                print("Aici:",i,j,epsilon,abs(m1[i][j]-m2[i][j]))
                return ('Not equal')
    return ('Matrici egale')


def circuit_test(input, control_gate):
    #circuit pentru prima poarta:
    circuit=QuantumCircuit(4,4)

    circuit=circuit.compose(input)
    circuit.barrier()

    poarta=QuantumCircuit(4,4)
    poarta=poarta.compose(control_gate.control(3,None,'010'))
    #print(poarta.draw())
    
    #inainte de masuratori, altfel nu merge simulatorul unitary:
    simulator=Aer.get_backend('unitary_simulator')
    result=execute(poarta,simulator).result()
    matrice1=result.get_unitary(poarta)
    #print("Matricea primului circuit este:\n",matrice1)
    print("")

    circuit=circuit.compose(poarta)
    circuit.barrier()

    circuit.measure(0,0)
    circuit.measure(1,1)
    circuit.measure(2,2)
    circuit.measure(3,3)

    print(circuit.draw())

    #circuit pentru a doua poarta:
    circuit2=QuantumCircuit(4,4)

    circuit2=circuit2.compose(input)
    circuit2.barrier()

    poarta=QuantumCircuit(4,4)
    
    control_gate=control_gate.power(3)
    poarta.x(0)
    poarta.x(2)
    poarta=poarta.compose(MCMT(control_gate,3,1))
    poarta.x(0)
    poarta.x(2)
    
    #print(poarta.draw())

    simulator=Aer.get_backend('unitary_simulator')
    result=execute(poarta,simulator).result()
    matrice2=result.get_unitary(poarta)
    #print("Matricea celui de-al doilea circuit este:\n",matrice2)
    print("")


    circuit2=circuit2.compose(poarta)
    circuit2.barrier()
    

    print("")
    check=egalitate_matrici(matrice1,matrice2)
    if check=='Not equal':
        global raspuns
        raspuns=0
        #print("Haide bre",raspuns)
    print("Comparare matrici: ",check)
    print("")

    circuit2.measure(0,0)
    circuit2.measure(1,1)
    circuit2.measure(2,2)
    circuit2.measure(3,3)

    print(circuit2.draw())

    simulator=Aer.get_backend('qasm_simulator')
    result=execute(circuit,simulator,shots=1024).result()
    counts=result.get_counts()
    plot_histogram(counts).savefig('ex1.1.png')
    print("Rezultat prima poarta:")
    print(counts)
    result=execute(circuit2,simulator,shots=1024).result()
    counts=result.get_counts()
    plot_histogram(counts).savefig('ex1.2.png')
    print("Rezultat a doua poarta:")
    print(counts)



exemplu=QuantumCircuit(4,4)
exemplu.x(1)
exemplu.x(3)

circuit_test(exemplu, XGate())
circuit_test(exemplu, YGate())
circuit_test(exemplu, ZGate())
circuit_test(exemplu, HGate())
circuit_test(exemplu, SGate())

#print("De ce nu il ia",raspuns)

if raspuns==0:
    print("Portile nu sunt egale")
if raspuns==1:
    print("Portile sunt egale")

