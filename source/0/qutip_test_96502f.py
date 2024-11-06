# https://github.com/carstenblank/Quantum-Adder-Draper/blob/930adc6c3917334c156b84d94aa6470bc5b9932b/scribble/qutip_test.py
import sys
import os
from _ctypes import Array
from typing import Dict, List, Union

import math

import extensions.standard as gates
from extensions.standard import *
from extensions.standard.cx import CnotGate
from extensions.standard.h import HGate
from extensions.standard.u1 import U1Gate
from extensions.standard.u3 import U3Gate
from extensions.standard.x import XGate
from qiskit import QuantumProgram, QuantumCircuit, QuantumRegister, InstructionSet, Gate, CompositeGate

###############################################################
# Set the device name and coupling map.
###############################################################
device = "ibmqx2"
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}

###############################################################
# Make a quantum program for the GHZ and Bell states.
###############################################################
QPS_SPECS = {
    "name": "programs",
    "circuits": [{
        "name": "ghz",
        "quantum_registers": [{
            "name": "q",
            "size": 5
        }],
        "classical_registers": [
            {"name": "c",
             "size": 5}
        ]}
    ]
}


def phase(θ: float, q: QuantumRegister) -> List[Gate]:
    return [
        U1Gate(θ, q),
        XGate(q),
        U1Gate(θ, q),
        XGate(q)
    ]


def R_z(θ: float, q: QuantumRegister) -> List[Gate]:
    return [
        XGate(q),
        U1Gate(θ / 2, q),
        XGate(q),
        U1Gate(-θ / 2, q)]


def R_y(θ: float, q: QuantumRegister) -> List[Gate]:
    return [ U3Gate(θ, 0, 0, q) ]


def U(α: float, β: float, γ: float, δ: float, q: QuantumRegister) -> List[Gate]:
    result = []
    result.extend(R_z(δ, q))
    result.extend(R_y(γ, q))
    result.extend(R_z(β, q))
    result.extend(phase(α, q))
    return result


def cNot(control: QuantumRegister, target: QuantumRegister, couplingMap: Dict[int, List[int]]) -> List[Gate]:
    result = []

    direction = None

    _, control_qbit_no = control
    _, target_qbit_no = target
    targets = couplingMap[control_qbit_no]
    if target_qbit_no in targets:
        direction = 0

    controls = couplingMap[target_qbit_no]
    if control_qbit_no in controls:
        direction = 1

    if direction == 0:
        result.append(CnotGate(control, target))
    if direction == 1:
        result.append(HGate(control))
        result.append(HGate(target))
        result.append(CnotGate(control, target))
        result.append(HGate(control))
        result.append(HGate(target))

    if direction is None:
        raise AssertionError()

    return result


def cU(α: float, β: float, γ: float, δ: float, control: QuantumRegister, target: QuantumRegister,
       couplingMap: Dict[int, List[int]]) -> List[Gate]:

    A = []
    A.extend(R_y(γ / 2, target))
    A.extend(R_z(β, target))

    B = []
    B.extend(R_z(-(δ + β) / 2, target))
    B.extend(R_y(-γ / 2, target))

    C = []
    C.extend(R_z((δ - β) / 2, target))

    result = []
    result.extend(C)
    result.extend(cNot(control, target, couplingMap))
    result.extend(B)
    result.extend(cNot(control, target, couplingMap))
    result.extend(A)
    result.append(U1Gate(α, control))

    return result


def cR_k(k: int, control: QuantumRegister, target: QuantumRegister,
         couplingMap: Dict[int, List[int]]) -> List[Gate]:
    λ = 2 * math.pi / math.pow(2, k)
    return cR(λ, control, target, couplingMap)


def cR(λ: float, control: QuantumRegister, target: QuantumRegister,
       couplingMap: Dict[int, List[int]]) -> List[Gate]:
    α = λ / 2
    β = λ / 2
    γ = λ / 2
    δ = λ / 2
    return cU(α, β, γ, δ, control, target, couplingMap)


qp = QuantumProgram(specs=QPS_SPECS)
ghz: QuantumCircuit = qp.get_circuit("ghz")
q = qp.get_quantum_registers("q")
c = qp.get_classical_registers("c")

circuit = cR_k(1, q[0], q[1], coupling_map)

gate: Gate
for gate in circuit:
    ghz._attach(gate)

print(ghz.qasm())
