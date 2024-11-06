# https://github.com/daehiff/gate-optimisation/blob/4aa96e323687670626f4f9fa0cab18f1a85ce86e/evaluations.py
import os
import time

import numpy as np
import pyzx as zx
import qfast
import subprocess
import logging as log
#import mitm
from pytket._tket.passes import FullPeepholeOptimise, RebasePyZX, RebaseIBM
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
from pytket._tket.predicates import CompilationUnit
from qiskit import QuantumCircuit, execute, BasicAer

from qiskit.compiler import transpile
from SQIR.VOQC.interop import SQIR
from utils import zx_circuit_to_qiskit_circuit, CLIFFORD_T_SET, qiskit_circuit_to_zx_circuit, to_qc_format

UNITARY_BACKEND = BasicAer.get_backend('unitary_simulator')


def rebase_gates_ibm(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Helper function to rebase the circuits to IBM gate set

    :param circ:
    :return:
    """
    cu = CompilationUnit(circuit_from_qasm_str(circ.qasm()))
    rebase_ibm = RebaseIBM()
    rebase_ibm.apply(cu)
    circ_out = QuantumCircuit.from_qasm_str(circuit_to_qasm_str(cu.circuit))
    return circ_out


def mitm_evaluation(circ: QuantumCircuit) -> QuantumCircuit:
    # unitary = qiskit_circuit_to_zx_circuit(circ).to_matrix().astype(dtype=np.complex128)
    # start = time.time()
    # qc_string = mitm.mitm_algorithm(unitary, 15, 1, True)
    # print(time.time() - start)
    # if qc_string == "":
    #    return None
    # circ = zx_circuit_to_qiskit_circuit(zx.Circuit.from_qc(qc_string))
    return circ


def tket_evaluation(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Evaluation of the cirquit optimizer by tket
    :param circ:
    :return:
    """
    log.warning("Evaluating tket")
    tketpass = FullPeepholeOptimise()
    refine_circs = RebasePyZX()
    cu = CompilationUnit(circuit_from_qasm_str(circ.qasm()))
    tketpass.apply(cu)
    refine_circs.apply(cu)
    res_circ = zx.Circuit.from_qasm(circuit_to_qasm_str(cu.circuit)).split_phase_gates()
    circ_out = zx_circuit_to_qiskit_circuit(res_circ)
    log.warning("Done!")
    return circ_out


def qiskit_evaluation(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Evaluation of the QISKIT optimizer
    :param circ:
    :return:
    """
    log.warning("Evaluating qiskit")
    circuit_out = transpile(circ, basis_gates=list(CLIFFORD_T_SET))
    assert verify_equality(circ, circuit_out)
    log.warning("Done")
    return circuit_out


def tpar_evaluation(circ: QuantumCircuit, root_dir: str = "") -> QuantumCircuit:
    """
    Evaluation of the t-par algorithm by Amy et. al.

    :param root_dir: see below
    :param circ:
    :return:
    """
    log.warning("Evaluating t-par")
    out = to_qc_format(circ, replace_s=True)
    with open("temp.qc", "w") as f:
        f.write(out)

    result = subprocess.run(root_dir + "t-par/t-par < temp.qc", stdout=subprocess.PIPE, shell=True)
    result.check_returncode()
    os.remove("temp.qc")
    result = result.stdout.decode("utf-8")
    if "ERROR" in result:
        raise Exception(f"t-par couldn't parse the circuit: {result}")
    zx_circ = zx.Circuit.from_qc(result).to_basic_gates().split_phase_gates()
    circ_out = zx_circuit_to_qiskit_circuit(zx_circ)
    #    assert verify_equality(circ, circ_out)
    log.warning("done")
    return circ_out


def voqc_evaluation(circ: QuantumCircuit, root_dir: str = "") -> QuantumCircuit:
    """
    Executes VOQC by its python wrapper
    :param root_dir: speific relative position from . to rootdir
    :param circ:
    :return:
    """
    log.warning("Evaluating voqc")
    with open(root_dir + "SQIR/VOQC/temp.qasm", "w") as f:
        f.write(circ.qasm())
    out_circ = QuantumCircuit.from_qasm_str(SQIR(fname="temp.qasm").optimize().write_str()).decompose()
    out_circ = qiskit_circuit_to_zx_circuit(out_circ).to_basic_gates().split_phase_gates()
    out_circ = zx_circuit_to_qiskit_circuit(out_circ)
    assert verify_equality(circ, out_circ)
    os.remove(root_dir + "SQIR/VOQC/temp.qasm")
    log.warning("Done")
    return out_circ


def q_fast_evaluation(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Evaluation of the Q-Fast algorithm using Qskits Aer to create the unitary matrix
    * QFast is an approximate tool hence, its not granted to find, the correct out circuit, but a unitary, that is close
        => we will not assert this here
    * QFast doesn't allow natively to adjust the set of gates and produces circuits, that include the u/u3 gate
        by IBM. They claim, that this gate is one of their basic instructions, hence I would assume this as also
        part of the clifford+T group.

    :param circ:
    :return:, 0
    """
    log.warning("Evaluating Qfast")
    unitary = qiskit_circuit_to_zx_circuit(circ).to_matrix().astype(dtype=np.complex128)
    circ_qasm = qfast.synthesize(unitary)
    circ_ = QuantumCircuit.from_qasm_str(circ_qasm)
    log.warning("Done")
    return circ_


def pyzx_evaluation(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Evaluation using the pyzx libary.

    due to compability issues, it needs to be casted from QSkits Circuit to the internal datastrcuture
    => check if this consumes to much memory
    :param circ:
    :return:
    """
    log.warning("Evaluating pyzx")
    c_graph = qiskit_circuit_to_zx_circuit(circ).to_graph()
    zx.simplify.full_reduce(c_graph)
    cir_reduced = zx.extract_circuit(c_graph).split_phase_gates().to_basic_gates()
    circ_out = zx_circuit_to_qiskit_circuit(cir_reduced)
    # circ_out = convert_clifford_t(circ_out)
    assert verify_equality(circ, circ_out)
    log.warning("done")
    return circ_out


def circuit_to_unitary(qc: QuantumCircuit) -> np.array:
    """
    Converts any given QuantumCircuit into a unitary matrix
    Currently hacky way using AER backend.
    :param qc:
    :return:
    """
    job = execute(qc, UNITARY_BACKEND)
    res = job.result().get_unitary(qc, decimals=15)
    return res


def verify_equality(qc_in: QuantumCircuit, qc_out: QuantumCircuit):
    qc_in_ = qiskit_circuit_to_zx_circuit(qc_in)
    qc_out_ = qiskit_circuit_to_zx_circuit(qc_out)
    return qc_in_.verify_equality(qc_out_)  # and is_clifford_t(qc_out)
