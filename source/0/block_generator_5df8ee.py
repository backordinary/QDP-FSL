# https://github.com/Baccios/CTC_iterator/blob/5b47d56702373edd9d928118f721475f676fbd0b/ctc/block_generator.py
"""
This module provides an API to generate parametrized CTC assisted circuits.
With CTC circuit we mean a gate used for an interaction with a
Deutschian Closed Timelike Curve.
"""
from math import pi

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions import UnitaryGate

from ctc.brun import get_u_qiskit


def _generate_v_circuit_nbp(size):
    """
    Get a CTC gate using the recipe in
    <a href="https://arxiv.org/abs/1901.00379">this article</a>

    :param size: Size (in qubits) of the gate instance
    :type size: int
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """
    # build the V sub circuit
    ctc_circuit = QuantumCircuit(2 * size, name='V Gate')

    # ### R block

    for i in range(size):
        ctc_circuit.cu(-pi / 2 ** i, 0, 0, 0, i, size)

    # ### T block

    for i in range(size + 1, 2 * size):
        ctc_circuit.ch(size, i)

    # ### W block

    for i in range(size + 1, 2 * size):
        ctc_circuit.cu(pi / size, 0, 0, 0, i, size)

    # ### C block

    for i in range(size):
        ctc_circuit.cnot(i, size + i)

    # return the result
    return ctc_circuit


def _generate_v_circuit_brun_fig2(size=2):
    """
    Get a CTC gate using the recipe in Fig. 2 of
    <a href="https://arxiv.org/abs/0811.1209">this article</a>

    :param size: Size (in qubits) of the gate instance (only value of 2 is implemented for this method)
    :type size: int
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """

    if size != 2:
        raise ValueError("Error. This algorithm has only been implemented with a size n=2. Use \"brun\" instead.")

    u00 = QuantumCircuit(2, name="U00")
    u00.swap(0, 1)

    u01 = QuantumCircuit(2, name="U01")
    u01.x(0)
    u01.x(1)

    u10 = QuantumCircuit(2, name="U10")
    u10.h(0)
    u10.x(0)

    u11 = QuantumCircuit(2, name="U11")
    u11.swap(0, 1)
    u11.x(0)
    u11.h(1)

    cc_u00 = u00.control(2, ctrl_state='00')
    cc_u01 = u01.control(2, ctrl_state='10')
    cc_u10 = u10.control(2, ctrl_state='01')
    cc_u11 = u11.control(2, ctrl_state='11')

    qr = QuantumRegister(4)
    ctc_circuit = QuantumCircuit(qr, name="V_gate")
    ctc_circuit.append(cc_u00.to_instruction(), qr[0:4])
    ctc_circuit.append(cc_u01.to_instruction(), qr[0:4])
    ctc_circuit.append(cc_u10.to_instruction(), qr[0:4])
    ctc_circuit.append(cc_u11.to_instruction(), qr[0:4])

    # return the result
    return ctc_circuit


def _generate_v_circuit_brun(size, two_dim=True, section_divider=None):
    """
    Get a CTC gate using the general recipe in
    <a href="https://arxiv.org/abs/0811.1209">this article</a>

    :param size: Size (in qubits) of the gate instance
    :type size: int
    :param two_dim: if set to True, the encoding considered is |psi_k> = cos(k*pi/2^n)|0> + sin(k*pi/2^n)|1>, otherwise
    it is the 3d encoding scheme. Defaults to True.
    :type two_dim: bool
    :param section_divider: If two_dim is True, the statevector representation will be
    |psi_k> = cos(k*pi/sector_divider)|0> + sin(k*pi/sector_divider)|1>. If None, it will be considered as 2^size.
    :type section_divider: int
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """

    def get_str(k):
        return format(k, '0' + str(size) + 'b')

    qr = QuantumRegister(2*size)
    ctc_circuit = QuantumCircuit(qr, name="V_gate")

    for k in range(2**size):
        u_k = UnitaryGate(get_u_qiskit(k, size, two_dim, section_divider), label="U_" + get_str(k))
        cu_k = u_k.control(size, label="CU_" + get_str(k), ctrl_state=get_str(k)[::-1])
        # print("Straight ", k, " = ", get_str(k))
        # print("Reverse ", k, " = ", get_str(k)[::-1])  # DEBUG
        ctc_circuit.append(cu_k, qr[0:2**size])

    return ctc_circuit


def get_ctc_assisted_circuit(size, method="nbp"):
    """
    Get a CTC gate specifying its size and (optionally) the method used to build it.

    :param size: The size (in qubits) of the gate.
                 The resulting gate will have 2*size qubits because
                 the first half represents the CTC
                 and the second half represents the Chronology Respecting (CR) system.
    :type size: int
    :param method: the algorithm used to build the gate. It defaults to "nbp".
    for a list of possible values refer to simulation.CTCCircuitSimulator documentation
    :type method: str
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """
    # Other methods are left for future updates
    if method == "nbp":
        return _generate_v_circuit_nbp(size)
    elif method == "brun_fig2":
        return _generate_v_circuit_brun_fig2(size)
    elif method == "brun":
        return _generate_v_circuit_brun(size)
    elif method == "brun_3d":
        return _generate_v_circuit_brun(size, two_dim=False)
    elif method == "brun_quadrant":
        return _generate_v_circuit_brun(size, two_dim=True, section_divider=2**(size + 2))
    else:
        raise ValueError("method must be set to one of the specified values")
