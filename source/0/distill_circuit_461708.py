# https://github.com/vipasu/magic-distillation/blob/987853002c85b29900217de8a5498f97291aa6ca/distill_circuit.py
import qiskit
import numpy as np
import qiskit.quantum_info as qi
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.states import statevector
import qiskit.circuit.library as qlib
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit import Aer, execute


I = [[1,0],[0, 1]]
Z = qi.Pauli.pauli_single(1, 0, "Z").to_matrix()
Y = qi.Pauli.pauli_single(1, 0, "Y").to_matrix()
X = qi.Pauli.pauli_single(1, 0, "X").to_matrix()
N = 5


beta = 1/2 * np.arccos(1/np.sqrt(3))
c = np.cos(beta)
s = np.sin(beta)
t0 = statevector.Statevector(np.array([c, np.exp(1j * np.pi/4) * s]))
t1 = statevector.Statevector(np.array([s, -np.exp(1j * np.pi/4) * c]))
zero_state = statevector.Statevector(np.array([1, 0]))
one_state = statevector.Statevector(np.array([0, 1]))


def distill_circuit():
    # construct circuit
    circuit = qiskit.QuantumCircuit(N)
    # layer 1
    circuit.cz(3, 0)
    circuit.cz(3, 2)
    circuit.cx(3, 4)
    circuit.cz(3, 4)



    # layer 2
    circuit.cz(2, 0)
    circuit.cz(2, 1)
    circuit.cx(2, 4)
    # layer 3
    circuit.cx(1, 4)
    # layer 4
    circuit.cx(0, 4)
    circuit.cz(0, 4)


    # layer 5
    circuit.z(0)
    circuit.z(3)
    circuit.z(4)

    circuit.h(0)
    # circuit.x(1)
    circuit.h(1)
    circuit.h(2)
    circuit.h(3)
    circuit.h(4)

    circuit.y(4)
    return circuit


# In[42]:


bk_circuit = distill_circuit()

def distill(state, post='0000'):
    output = state.evolve(bk_circuit)
    while True:
        string, dm = output.measure(range(4)) # measure first four bits
        if string == post:
            break
    return dm

def magic_fidelity(dm, direction=None):
    if direction == 0:
        return qi.state_fidelity(t0, dm)
    elif direction == 1:
        return qi.state_fidelity(t1, dm)
    else:
        if not dm.is_valid():
            print(dm)
        return max(qi.state_fidelity(t0, dm), qi.state_fidelity(t1, dm))

def compare_fidelity(initial, final):
    init_dm = qi.partial_trace(initial, range(4))
    final_dm = qi.partial_trace(final, range(4))
    M_init = magic_fidelity(init_dm)
    M_final = magic_fidelity(final_dm)
    F_init = qi.state_fidelity(t0, init_dm)
    F_final = qi.state_fidelity(t0, final_dm)
    print("Initial T fidelity: ", qi.state_fidelity(t0, init_dm))
    print("Final T fidelity: ", qi.state_fidelity(t0, final_dm))
    return F_init, F_final


def tensor_n(state, n):
    new_state = state.copy()
    for i in range(n-1):
        new_state = new_state.tensor(state)
    return new_state

def repeat_distill(state, n=1, full=False, verbose=True, return_qubit=False, direction=None):
    if not full:
        new_in = tensor_n(state, 5)
    else:
        new_in = state
    new_out = distill(new_in)
    new_out_qubit = qi.partial_trace(new_out, range(4))
    fid = magic_fidelity(new_out_qubit, direction=direction)
    if return_qubit:
        ret = new_out_qubit
    else:
        ret = fid

    if verbose:
        if direction:
            print("T_{direction} fidelity: ",fid)
        else:
            print("Maximum T fidelity: ", fid)


    if n == 1:
        return [ret]
    else: # return list of all results
        final = [ret]
        final.extend(repeat_distill(new_out_qubit, n-1, full=False, verbose=verbose, return_qubit=return_qubit, direction=direction))
        return final

def successful_distillation(state, n, full=False, verbose=False):
    fid = repeat_distill(state, n, full, verbose)
    if fid >= .99:
        return True
    else:
        print(fid)
        return False


def create_ghz_t(alpha, n):
    beta = np.sqrt(1 - alpha**2)
    return alpha * tensor_n(t0, n) + beta * tensor_n(t1, n)


def create_ghz_phase(alpha, n):
    beta = np.sqrt(1 - alpha**2)
    return alpha * tensor_n(zero_state, n) + np.exp(1j * np.pi/4) * beta * tensor_n(one_state, n)

def hypergraph_to_circuit(hg, n):
    circuit = qiskit.QuantumCircuit(n)
    for i in range(n): # initialize to all |+> state
        circuit.h(i)

    for h_edge in hg:
        # Apply C^k Z
        k = len(h_edge)
        if k > 1:
            circuit.append(qlib.ZGate().control(k-1), h_edge)
        else: # a single node hyperedge is a z_gate
            circuit.z(h_edge[0])
    return circuit

def hypergraph_to_state(hg, n):
    """Create a hypergraph state from input hg (set of hyperedges)
    hg should be a list of lists.
    """
    circ = hypergraph_to_circuit(hg, n)
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(circ, simulator).result()
    return statevector.Statevector(result.get_statevector())
