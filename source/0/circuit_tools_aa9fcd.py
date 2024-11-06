# https://github.com/TimWeaving/Qondense/blob/346d86254817312fc68a481be592bd998b14c5dc/qondense/utils/circuit_tools.py
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

# TODO update exp_P to use cascade function
def cascade(q_index, num_qubits=None, circ=None, reverse=False):
    cascade_bits = sorted(q_index)

    if circ is None:
        assert(num_qubits is not None)
        circ = QuantumCircuit(num_qubits)
    else:
        num_qubits = circ.num_qubits

    if reverse:
        cascade_bits.reverse()
        for q in range(len(cascade_bits)-1):
            circ.cx(cascade_bits[q+1], cascade_bits[q])
    else:
        for q in range(len(cascade_bits)-1):
            circ.cx(cascade_bits[q], cascade_bits[q+1])

    return circ


def exp_P(p_string, control=None, circ=None, rot=0):
    """Exponentiate a Pauli string given a rotation

    Parameters
    ----------
    p_string: str
        Pauli string P to be exponentiated
    control: int optional
        specified control qubit
    circ: QuantumCircuit
        Append gates to an existing QuantumCircuit
    rot: float optional
        the rotation r in e^(irP)

    Returns
    -------
    QuantumCircuit
    """
    num_qubits = len(p_string)
    
    # index X, Y, Z in the string of Paulis
    p_index = {}
    for index, p in enumerate(p_string):
        q_pos = num_qubits-1-index
        if p not in p_index:
            p_index[p] = [q_pos]
        else:
            p_index[p] += [q_pos]

    # initiate quantum circuit object
    if circ is None:
        circ = QuantumCircuit(num_qubits)
        #circ.barrier()
    
    # rotate X and Y Paulis into Z basis
    for q in p_index.keys():
        if q == 'X':
            #rotate X to Z
            for i in p_index['X']:
                circ.h(i)
            #circ.barrier()
        
        elif q == 'Y':
            #rotate Y to Z
            for i in p_index['Y']:
                circ.sdg(i)
                circ.h(i)
            #circ.barrier()
            
        else:
            pass
    
    # evaluate parity of remaining Z qubits
    if 'I' in p_index:
        #return blank circuit if all qubits are identity
        if p_index['I'] == list(range(num_qubits)):
            return circ
        #Index qubits which are non identity
        else:
            non_I = list(set(range(num_qubits)) - set(p_index['I']))
            num_Z = range(len(non_I)-1)
    else:
        non_I = list(range(num_qubits))
        num_Z = range(num_qubits - 1)
    
    non_I = sorted(non_I)

    # cascade of CNOT gates between adjacent non-identity qubits
    for i in num_Z:
        circ.cx(non_I[i], non_I[i+1])
    
    # apply the rotation
    if control is None:
        circ.rz(-2*rot, non_I[-1])
    else:
        circ.crz(-2*rot, control, non_I[-1])
        
    # reverse cascade of CNOT gates between adjacent non-identity qubits
    for i in num_Z:
        circ.cx(non_I[len(num_Z)-i-1], non_I[len(num_Z)-i])
        
    # rotate X and Y Paulis into Z basis
    for q in p_index.keys():
        if q == 'X':
            #rotate X to Z
            for i in p_index['X']:
                circ.h(i)
            #circ.barrier()
        
        elif q == 'Y':
            #rotate Y to Z
            for i in p_index['Y']:
                circ.h(i)
                circ.s(i)
            #circ.barrier()
            
        else:
            pass
        
    #circ.barrier()
    
    return circ


def circ_from_paulis(init_state=[], paulis=[], params=[], rots=[], circ=None, trot_order=1, dup_param=False):
    """Exponentiate a list of Pauli operators and trotterize

    Paramaters
    ----------
    init_state: list optional
        qubit positions that should have value 1 (apply X). By default all 0.
    paulis: list optional
        Pauli strings, applied left to right
    params: list optional
        angles by which to rotate each exponentiated Pauli, leave empty to fill with parameters for optimisation in VQE
    rots: list optional
        rotations from CS-VQE, applied left to right

    Returns
    -------
    QuantumCircuit
    """
    
    # parameters to be optimised in VQE routine
    #if params == []:
    #    param_chars = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','ς','σ','τ','υ','φ','χ','ψ','ω']
    #    params = [] 
    #    for comb in list(itertools.combinations(param_chars, 3)):
    #        char_str = ''.join(comb)
    #        params.append(Parameter(char_str))
    if params == []:
        params = ParameterVector('P', len(paulis))

    #initiate quantum state (usually Hartree Fock)
    if circ is None:
        circ = QuantumCircuit(len(paulis[0]))
        for q in init_state:
            circ.x(q)

    # applies the ansatz
    for t in range(trot_order):
        for index, p in enumerate(paulis):
            circ += exp_P(p_string = p, rot = params[index]/trot_order)
    
    # rotates in accordance with CS-VQE routine
    for r in rots:
        circ += exp_P(p_string = r[1], rot = r[0])
      
    return circ


def cancel_pairs(circ, hit_set):
    """ Cancel neighbouring inverse gates, eg. S and Sdg, or H^2
    """
    gate_dict={}
    for i in range(circ.num_qubits):
        gate_dict[i] = []
    for index, gate in enumerate(circ.data):
        gate_dict[gate[1][0].index].append((index, gate[0].name))
    delete_index = []
    for i in gate_dict.keys():
        gates = gate_dict[i]
        for j in range(len(gates)-1):
            pair = list(zip(*(gates[j], gates[j+1])))
            if set(pair[1]) == hit_set:
                for k in pair[0]:
                    delete_index.append(k)
    delete_index = sorted(delete_index)

    shift=0
    for i in delete_index:
        circ.data.pop(i-shift)
        shift+=1