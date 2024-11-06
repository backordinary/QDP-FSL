# https://github.com/Jabi7/Qstuff/blob/415ac927c3ff04701edf4ebf78dbd3e8868e44b4/2a.py
# The starting pattern is represented by this list of numbers.
# Please use it as an input for your solution.
lights = [0, 1, 1, 1, 0, 0, 1, 1, 1]

def week2a_ans_func(lights):
    ##### build your quantum circuit here
    from qiskit import IBMQ, Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
    
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
    
    v = QuantumRegister(9, name='v')
    c = QuantumRegister(9, name='c')
    o= QuantumRegister(1, name='out')
    cbits = ClassicalRegister(9, name='cbits')
    qc = QuantumCircuit(v, c, o, cbits)

    qc.x(o[0])
    qc.h(o[0])

    qc.h(v)
        
    for i in range(len(lights)):
        if lights[i]==1:
            qc.x(c[i])
    qc.barrier()
    for i in range(12):    
        qc.cx(v[0], [c[0],c[1],c[3]])
        qc.cx(v[1], [c[1],c[0],c[2],c[4]])
        qc.cx(v[2], [c[2],c[1],c[5]])
        qc.cx(v[3], [c[3],c[0],c[4], c[6]])
        qc.cx(v[4], [c[4],c[1],c[3],c[5],c[7]])
        qc.cx(v[5], [c[5],c[2],c[4],c[8]])
        qc.cx(v[6], [c[6],c[3],c[7]])
        qc.cx(v[7], [c[7],c[6],c[4],c[8]])
        qc.cx(v[8], [c[8],c[7],c[5]])
        qc.mct(c, o)
        qc.cx(v[0], [c[0],c[1],c[3]])
        qc.cx(v[1], [c[1],c[0],c[2],c[4]])
        qc.cx(v[2], [c[2],c[1],c[5]])
        qc.cx(v[3], [c[3],c[0],c[4], c[6]])
        qc.cx(v[4], [c[4],c[1],c[3],c[5],c[7]])
        qc.cx(v[5], [c[5],c[2],c[4],c[8]])
        qc.cx(v[6], [c[6],c[3],c[7]])
        qc.cx(v[7], [c[7],c[6],c[4],c[8]])
        qc.cx(v[8], [c[8],c[7],c[5]])

        qc.append(diffuser(9), [0,1,2,3,4,5,6,7,8])

        qc.barrier()
    qc.measure(v, cbits)
    qc = qc.reverse_bits()
# def puzz(qc, 

    
    #####  In addition, please make it a function that can solve the problem even with different inputs (lights). We do validation with different inputs.
    
    return qc
