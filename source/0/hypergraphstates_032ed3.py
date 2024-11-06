# https://github.com/quantcert/quantcert.github.io/blob/369089e3319ead9947abb3694f3ad64dac255501/Mermin-hypergraph-states/Qiskit/app/mermin_on_qiskit/hypergraphstates.py
from qiskit import *


def circuit_creation(n, hyperedges):
    """ Creates an empty circuit with the number of qubits required.

    :param int n: The number of qubits of which depends the number of wires to 
        create.
    :param list[list[int]] hyperedges: A list containing the lists of the 
        vertices which are linked by an hyperedge.
    :returns: QuantumCircuit -- A circuit with the required number of quantum 
        wires and classical wires.
    """
    created = False
    i = 0
    qubits = 0
    while not created and i < len(hyperedges):
        hyperedge = hyperedges[i]
        if len(hyperedge) > 3:
            qubits = 2 * n - 2
            created = True
        i += 1
    if not created:
        qubits = n
    circuit = QuantumCircuit(qubits, n)
    return circuit


def circuit_initialisation(n, hyperedges):
    r""" Creates an empty circuit with the number of qubits required and places 
        the circuit in the initial state before adding gates for calculations.
        Places and Hadamard gate on every main (non additional) qubits wire.
        This is needed in order to place the qubits in a `|+>` state which is
        `(|0> + |1>) / sqrt(2)`.

    :param int n: The number of qubits of which depends the number of wires to 
        create.
    :param list[list[int]] hyperedges: A list containing the lists of the 
        vertices which are linked by an hyperedge.
    :returns: QuantumCircuit -- The created and initialized circuit.

    """
    circuit = circuit_creation(n, hyperedges)
    circuit.h(range(n))
    return circuit


def edges_layout(n, hyperedges, circuit):
    """ Disposes all the gates corresponding to the edges.
        In fact, for a two-vertices edge, there not much to do, as for a 
        three-vertices edge, too.
        For more an edge of than three vertices, things are a little different. 
        First, Toffoli gates are used to link qubits
        two by two. The target qubit here is an auxiliary qubit.
        Then, the last link is a simple CZ.
        But all the Toffoli gates that we put create an entanglement which is 
        removed by replacing exactly the same gates again after the CZ.

    :param int n: The number of qubits.
    :param list[list[int] hyperedges: The list of the vertices which are linked
        by an hyperedge.
    :param QuantumCircuit circuit: The circuit that will be modified.
    :return: None
    """
    for hyperedge in hyperedges:
        if len(hyperedge) == 2:
            circuit.cz(hyperedge[0], hyperedge[1])
        if len(hyperedge) == 3:
            # The target qubit is the third one
            circuit.h(hyperedge[2])  
            circuit.toffoli(*hyperedge)
            circuit.h(hyperedge[2])
        if len(hyperedge) > 3:
            # Here, the process is different. We'll place many Toffoli gates to 
            # form the hypergraph-state which creates an entanglement. So, we 
            # place new Toffoli gates again for cancel the entanglement
            vertex_index = 0
            maximum = len(hyperedge) - 1
            qubit_aux = n
            circuit.toffoli(hyperedge[vertex_index], hyperedge[vertex_index + 1], qubit_aux)
            vertex_index += 2
            while vertex_index < maximum:
                circuit.toffoli(hyperedge[vertex_index], qubit_aux, qubit_aux + 1)
                qubit_aux += 1
                vertex_index += 1
            circuit.cz(qubit_aux, hyperedge[vertex_index])
            qubit_aux -= 1
            vertex_index -= 1
            while maximum > 2:
                circuit.toffoli(qubit_aux - 1, hyperedge[vertex_index], qubit_aux)
                qubit_aux -= 1
                vertex_index -= 1
                maximum -= 1
            circuit.toffoli(hyperedge[vertex_index], hyperedge[vertex_index - 1], qubit_aux)
