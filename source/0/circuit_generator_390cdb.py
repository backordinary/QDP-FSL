# https://github.com/quantcert/quantcert.github.io/blob/369089e3319ead9947abb3694f3ad64dac255501/Testing_contextuality/Src/codes_W5_2/Circuit_generator.py
from qiskit import *
from W_5_2 import *

# The functions XYZmeasurement add the necessary gates on the wire q to perform a measurement on the auxilary qubit qmeas
# When the measurement implies a change of basis (X and Y) measurement, the counter rotation is applied after measurement

def Xmeasurement(qc,q,qmeas):
    qc.h(q)
    qc.cx(q,qmeas)
    qc.h(q)

def Zmeasurement(qc,q,qmeas):
    qc.cx(q,qmeas)

def Ymeasurement(qc,q,qmeas):
    qc.sdg(q)
    qc.h(q)
    qc.cx(q,qmeas)
    qc.h(q)
    qc.s(q)

# The function circuit_create add to the quantum circuit the gates corresponding to  a measurement defined by tab
# tab is an array of size three containing integer from 0 to 3. Each integer corresponds to a measurement 0=I, 1=X,2=Y, 3=Z
# for instance if tab=[0,1,2] the function will generate a circuit corresponding to the IXZ measurement

def circuit_create(qc,tab,qmeas):
    if tab[0]==1:
        Xmeasurement(qc,0,qmeas)
    if tab[0]==2:
        Ymeasurement(qc,0,qmeas)
    if tab[0]==3:
        Zmeasurement(qc,0,qmeas)
        
    if tab[1]==1:
        Xmeasurement(qc,1,qmeas)
    if tab[1]==2:
        Ymeasurement(qc,1,qmeas)
    if tab[1]==3:
        Zmeasurement(qc,1,qmeas)
        
    if tab[2]==1:
        Xmeasurement(qc,2,qmeas)
    if tab[2]==2:
        Ymeasurement(qc,2,qmeas)
    if tab[2]==3:
        Zmeasurement(qc,2,qmeas)
        
# The function circuit_context_create concatanates three measurements corresponding to a context. 
# The context is given by an array of array "line" which is made of 3 size two vector. The lines are computed in the W5_2 script. 
# To separate the three measurements a barrier is added between each measurement
# Example: if line=[[1,2,3],[0,2,3],[1,0,0]] the function circuit_context_create will call 3 times the function circuit_create to concatanate the measurements XZY-IZY-XII

def circuit_context_create(qc,line,qmeas):
    circuit_create(qc,line[0],qmeas)
    qc.barrier()
    circuit_create(qc,line[1],qmeas)
    qc.barrier()
    circuit_create(qc,line[2],qmeas)
    qc.barrier()

# The function circuit_measure_context excecute the calculation correponding to the measurement of a contexte given by "line" either on the simulator or on some IBMQ machine
# it returns the expectation of the measurement and its standard deviation

def circuit_measure_context(qc,line,qmeas,number_of_shots, is_simulation=False):
    qc=QuantumCircuit(n+1,1)
    circuit_context_create(qc,line,qmeas)
    qc.measure([n], [0])
    if is_simulation:
        local_simulator = Aer.get_backend('qasm_simulator')
        result = execute(qc, backend=local_simulator, shots=number_of_shots).result()
    else:
        IBMQ.load_account()
        provider=IBMQ.get_provider('ibm-q-research')
        quantum_comp=provider.get_backend('ibmq_casablanca')
        result = execute(qc, backend=quantum_comp, shots=number_of_shots).result()
    measures_dictionary = result.get_counts()
    print(measures_dictionary)
    value_0=0
    value_1=0
    for measure in measures_dictionary:
        number_of_1=measure.count('1')
        if number_of_1==0:
            value_0+=measures_dictionary[measure]
        else:
            value_1+=measures_dictionary[measure]
        expect=(value_0-value_1)/number_of_shots
        print(expect)
        p = value_1 / number_of_shots
        var = p * (1 - p) / number_of_shots
        res = [expect, var]
    return(res)

def circuit_mermin(qc,All_lines,qmeas):
    C=0
    Var=0
    for Line in All_lines:
        print(Line)
        L=[]
        for i in range(3):
            p=Line[0][i]+1
            L.append(convert(p,4))
        print(L)
        res=circuit_measure_context(qc, L, 3, number_of_shots)
        #print(circuit_measure_context(qc, L, 3, number_of_shots))
        C += res[0] * Line[1]
        Var += res[1]
        print(C)
        print(np.sqrt(Var))

#Intialisation of the size of the circuit, the number of shots and the qubit to be measured
circuit=QuantumCircuit(4,1)
number_of_shots=8192
qmeas=3

# command to run the program, the contexts are encoded in All_lines which are computed in the file W5_2. There is a possibility to run the program on the set of contexts All_linesQ corresponding to the line of the hyperbolic quadric. Thoses lines are also computed in the python file W5_2
circuit_mermin(circuit,All_lines,qmeas)

#circuit_mermin(circuit,All_linesQ,qmeas)
