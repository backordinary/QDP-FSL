# https://github.com/f-fathurrahman/ffr-quantum-computing/blob/519c3e395380bf1602446dd2f000f81e36c6abc2/daniel_koch_lectures/our_qiskit_functions.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer

import math
import numpy as np

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

USING_QISKIT_CONVENTION = False

def toBinary(number, total):
    Nqubits = int( math.log2(total) )
    N = number
    b_num = np.zeros(Nqubits)
    for i in range(Nqubits):
        if N/( 2**(Nqubits-i-1)) >= 1:
            b_num[i] = 1
            N = N - 2**(Nqubits-i-1)
    B = []
    for j in range(Nqubits):
        B.append(int(b_num[j]))
    if not USING_QISKIT_CONVENTION:
        # reverse the list if not using Qiskit convention
        B.reverse()
    return B


def Wavefunction(obj, *args, **kwargs):
    #
    if type(obj) == QuantumCircuit:
        job = execute(obj, S_simulator, shots=1)
        statevec = np.asarray(job.result().get_statevector())
        # It is best to cast the result to np.array to get around deprecation
    #
    if type(obj) == np.ndarray:
        statevec = obj
    #
    #print("statevec = ")
    #print(statevec)
    #print()
    #
    is_sys = False
    new_line = False
    dec = 5
    #
    wavefunction = ""
    Nqubits = int(math.log2(len(statevec)))
    #
    for i in range(len(statevec)):
        # Round the coefficient
        value = round(statevec[i].real, dec) + round(statevec[i].imag, dec)*1j
        if (value.real != 0) or (value.imag != 0):
            state = list( toBinary(i, 2**Nqubits) )
            #print("state = ", state)
            state_str = ""
            # is_sys == true is skipped
            for j in range(len(state)):
                if type(state[j]) != str:
                    state_str += str(int(state[j])) # FIXME need int?
                else:
                    state_str += state[j]
            #
            #print("outside_loop: state_str = ", state_str)
            # Both real and imag components are not zero
            if (value.real != 0) and (value.imag != 0):
                if value.imag > 0:
                    wavefunction += str(value.real) + ' + ' + str(value.imag) + 'j |' + state_str + '>    '
                else:
                    wavefunction += str(value.real) + '' + str(value.imag) + 'j |' + state_str + '>    '
            #
            # Pure real case
            #
            if (value.real != 0) and (value.imag == 0):
                wavefunction += str(value.real) + ' |' + state_str + '>    '
            #
            # Pure imaginary case
            # 
            if (value.real == 0) and (value.imag != 0):
                wavefunction += str(value.imag) + 'j |' + state_str + '>    '
            if new_line:
                wavefunction += '\n'

    print(wavefunction)


def Measurement(qc, *args, **kwargs):
    #
    p_M = True
    S = 1
    ret = False
    NL = False
    #
    if "shots" in kwargs:
        S = int(kwargs["shots"])
    #
    if "return_M" in kwargs:
        ret = kwargs["return_M"]
    #
    if "print_M" in kwargs:
        p_M = kwargs["print_M"]
    #
    if "column" in kwargs:
        NL = kwargs["column"]

    print("S = ", S)

    M1 = execute(qc, M_simulator, shots=S).result().get_counts()
    M2 = {}
    k1 = list(M1.keys())
    v1 = list(M1.values())
    for k in range(len(k1)):
        key_list = list(k1[k])
        new_key = ""
        for j in range(len(key_list)):
            new_key = new_key + key_list[len(key_list) - (j+1)]
        M2[new_key] = v1[k]

    if p_M:
        k2 = list(M2.keys())
        v2 = list(M2.values())
        measurements = ""
        for i in range(len(k2)):
            m_str = str(v2[i]) + "|"
            for j in range(len(k2[i])):
                if k2[i][j] == "0":
                    m_str = m_str + "0"
                if k2[i][j] == "1":
                    m_str = m_str + "1"
                if k2[i][j] == " ":
                    m_str = m_str + ">|"
            m_str = m_str + ">    "
            if NL:
                m_str = m_str + "\n"
            measurements = measurements + m_str
        print(measurements)
        return
    
    if ret:
        return M2

    print("Should not pass here")


# qc will be modified
def blackbox_g_D(qc, qreg):
    f_type = ['f(0,1) -> (0,1)', 'f(0,1) -> (1,0)', 'f(0,1) -> 0', 'f(0,1) -> 1']
    idx_rand = np.random.randint(0, 4) # 4: exclusive
    #
    print("idx_rand = ", idx_rand)
    #
    if idx_rand == 0:
        qc.cx(qreg[0], qreg[1])

    if idx_rand == 1:
        qc.x(qreg[0])
        qc.cx(qreg[0], qreg[1])
        qc.x(qreg[0])
    
    if idx_rand == 2:
        qc.id(qreg[0])
        qc.id(qreg[1])
    
    if idx_rand == 3:
        qc.x(qreg[1])
    
    return f_type[idx_rand]


def deutsch(qc, qreg):
    qc.h(qreg[0])
    qc.h(qreg[1])
    f = blackbox_g_D(qc, qreg)
    qc.h(qreg[0])
    qc.h(qreg[1])
    return f


