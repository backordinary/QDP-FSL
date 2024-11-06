# https://github.com/ScoomenstheMumens/QiskitThesis-LGTQuantumSimulations/blob/e6aaad82ffa23b21b697b8c42c378ca15b4a11a2/lib/util.py
import warnings

import numpy as np
import qiskit
from qiskit import (ClassicalRegister, QuantumCircuit, QuantumRegister,
                    transpile)
from qiskit.circuit.random import random_circuit
from qiskit.ignis.verification.tomography import (StateTomographyFitter,
                                                  state_tomography_circuits)
from qiskit.quantum_info import Operator, state_fidelity
from qiskit.utils.mitigation.fitters import CompleteMeasFitter

#from lib import pauli_twirling
#from lib import convenience




def qiskit_calibration_circuits(N_qubits,qubits_measure=[0,2,4,6]):
    calib_circuits = []
    #state_labels = bin_list(N_qubits)
    N_qubits_measure=len(qubits_measure)
    state_labels1=bin_list(N_qubits_measure)
    for state in state_labels1:
        cr_cal = ClassicalRegister(N_qubits_measure, name = "c")
        qr_cal = QuantumRegister(N_qubits, name = "q_")
        qc_cal = QuantumCircuit(qr_cal, cr_cal, name=f"mcalcal_{state}")
        # prepares the state.
        for qubit in range(N_qubits_measure):
            if state[::-1][qubit] == "1":
                qc_cal.x(qr_cal[qubits_measure[qubit]])
        for j in range(N_qubits_measure):
            qc_cal.measure(qr_cal[qubits_measure[j]],cr_cal[j])
        calib_circuits.append(qc_cal)
    return calib_circuits, state_labels1

def GEM_calibration_circuits(qc,qubits_measure=[0,1,2,3,4]):
    '''
    returns the calibration circuits for the mitigation tecnique GEM.
    '''

    N_qubits_measure=len(qubits_measure)
    
    calib_circuits = [[],[]]
    N_qubits = len(qc.qubits)
    state_labels1 = bin_list(N_qubits_measure)
    
    qc_half_1, qc_half_2 = GEM_half_circuits(qc)
    # first half
    qr_1 = QuantumRegister(N_qubits, name="q")
    qc_cal_1 = QuantumCircuit(qr_1, name="cal_1")
    qc_cal_1.append(qc_half_1, qr_1)
    qc_cal_1.append(qc_half_1.inverse(), qr_1)
    
    qr_2 = QuantumRegister(N_qubits, name="q")
    qc_cal_2 = QuantumCircuit(qr_2, name="cal_2")
    qc_cal_2.append(qc_half_2, qr_2)
    qc_cal_2.append(qc_half_2.inverse(), qr_2)
    
    half_circuits = [qc_cal_1, qc_cal_2]
    for state in state_labels1:
        cr_cal = ClassicalRegister(N_qubits_measure, name = "c")
        qr_cal = QuantumRegister(N_qubits, name = "q_")
        qc_cal = QuantumCircuit(qr_cal, cr_cal, name=f"mcalcal_{state}")
        
        for qubit in range(N_qubits_measure):
            if state[::-1][qubit] == "1":
                qc_cal.x(qr_cal[qubits_measure[qubit]])
            # than we append the circuit
        qc_cal.append(half_circuits[0], qr_cal)

        for j in range(N_qubits_measure):
            qc_cal.measure(qr_cal[qubits_measure[j]],cr_cal[j])
        calib_circuits[0].append(qc_cal)
    for state in state_labels1:
        cr_cal = ClassicalRegister(N_qubits_measure, name = "c")
        qr_cal = QuantumRegister(N_qubits, name = "q_")
        qc_cal = QuantumCircuit(qr_cal, cr_cal, name=f"mcalcal_{state}")
        
        for qubit in range(N_qubits_measure):
            if state[::-1][qubit] == "1":
                qc_cal.x(qr_cal[qubits_measure[qubit]])
            # than we append the circuit
        qc_cal.append(half_circuits[1], qr_cal)

        for j in range(N_qubits_measure):
            qc_cal.measure(qr_cal[qubits_measure[j]],cr_cal[j])
        calib_circuits[1].append(qc_cal)
    return calib_circuits, state_labels1






def GEM_half_circuits(qc):
    '''
    this function splits the quantum circuit qc into two qauntum circuit:
    if the number of c_nots is even than it split equally, else the first pars has 
    1 c-not less than the second.
    '''
    try:
        N_cnots = qc.count_ops()["cx"]
    except:
        N_cnots = 0
    splitted_qasm = qc.qasm().split(";\n")
    splitted_qasm.remove("")
    half_1_qasm = ""
    half_2_qasm = ""
    i=0
    for j, element in enumerate(splitted_qasm):
        if "cx" in element:
            i+=1
        if j<3:
            half_1_qasm+=element+";\n"
            half_2_qasm+=element+";\n"
        else:
            if i<int(N_cnots/2+1):
                half_1_qasm+=element+";\n"
            else:
                if j!=len(splitted_qasm)-1:
                    half_2_qasm+=element+";\n"
    qc_half_1 = QuantumCircuit.from_qasm_str(half_1_qasm)
    #back_qc_half_1=transpile(qc_half_1,backend,optimization_level=0)
    qc_half_2 = QuantumCircuit.from_qasm_str(half_2_qasm)
    #back_qc_half_2=transpile(qc_half_2,backend,optimization_level=0)
    return qc_half_1, qc_half_2


####### fidelity and distance



######## utilities

def occurrences_to_vector(occurrences_dict):
    """Converts the occurrences dict to vector.
    Args:
    ----
        occurrences_dict (dict) : dict returned by BaseJob.results.get_counts() 
    
    Returns:
    ----
        counts_vector (np.array): the vector of counts.
    """
    counts_vector = np.zeros(2**len(list(occurrences_dict.keys())[0]))
    for state in list(occurrences_dict.keys()):
        counts_vector[int(state, 2)] = occurrences_dict[state]
    return counts_vector

def explicit_circuit(quantum_circuit):
    '''
    returns the explicit circuit of a 'quantum_circuit'
    
    splitted_qasm = quantum_circuit.qasm().split(";\n")
    qr_explicit = QuantumRegister(len(quantum_circuit.qubits), name="q")
    qc_explicit = QuantumCircuit(qr_explicit, name="qc")
    for element in splitted_qasm:
        if "cx" in element:
            el = element.replace("q[", "").replace("]", "")
            control_target = el.split(" ")[1].split(",") '''
    qc_qasm = quantum_circuit.qasm()
    return QuantumCircuit.from_qasm_str(qc_qasm)


    
def DecimalToBinary(num, number_of_qubits):
    """Converts a decimal to a binary string of length ``number_of_qubits``."""
    return f"{num:b}".zfill(number_of_qubits)

def bin_list(N_qubit):
    r=[]
    for i in range(2**N_qubit):
        r.append(DecimalToBinary(i,N_qubit))
    return r



