# https://github.com/marcobn/Quantum/blob/0b12d7ce39d589e0810c18f7b0eb183120c6ce38/BandTheory/tools.py
""" Utility functions without a place [yet] in qctb. """

import numpy as np
import qiskit

import qctb
import qctb.pyquiloperator

def hamiltonian(H, k):
    """ Same as operator Hamiltonian, but ungrouped to work with IQPE. """
    from pyquil.paulis import sI, sX, sY, sZ, ZERO

    Hk = H.Hk(k)
    pyop = ZERO()

    # PREPARE I AND Z GROUPS
    for α in range(H.M):
        h = float(Hk[α,α].real)
        if h == 0: continue
        pyop +=  h/2 * sI()
        pyop += -h/2 * sZ(α)

    # PREPARE X AND Y GROUPS
    for α in range(H.M-1):
        for β in range(α+1,H.M):
            h = float(Hk[α,β].real)
            if h == 0: continue
            pyop +=  h/2 * sX(α)*sX(β)
            pyop +=  h/2 * sY(α)*sY(β)

    # PREPARE XY GROUPS
    for α in range(H.M-1):
        for β in range(α+1,H.M):
            h = float(Hk[α,β].imag)
            if h == 0: continue
            pyop +=  h/2 * sY(α)*sX(β)
            pyop += -h/2 * sX(α)*sY(β)

    return qctb.pyquiloperator.reduce(pyop)

def evolution(op, τ=1, R=1):
    """ Evolve reciprocal operator in time with qiskit library. """
    # R *= int(np.ceil(τ))
    Ur = qiskit.aqua.operators.evolutions.PauliTrotterEvolution().convert(
        qiskit.aqua.operators.evolutions.Suzuki().convert(-2*np.pi*τ/R*op)
    ).to_circuit()
    U = qiskit.QuantumCircuit(Ur.num_qubits)
    for r in range(R):  U += Ur
    return U


# THIS VERSION IS DESIGNED FOR LINEAR ARCHITECTURE, BUT EMPIRICALLY...SUCKS
# def evolution(H, k, coef=1):
#     """ Evolve reciprocal operator in time with linear construction. """
#     def xxyy(θ):
#         """ Local evolution exp(iθ(XX+YY)) *maybe θ/2..? """
#         Ω = qiskit.QuantumCircuit(2)
#         Ω.h(0)
#         Ω.h(1)
#         Ω.cx(1,0)
#         Ω.rz(θ,0)
#         Ω.cx(1,0)
#         Ω.sxdg(0)
#         Ω.sxdg(1)
#         Ω.cx(1,0)
#         Ω.rz(θ,0)
#         Ω.cx(1,0)
#         Ω.h(0)
#         Ω.h(1)
#         Ω.s(0)
#         Ω.s(1)
#         return Ω.to_gate()
#
#     def yxxy(θ):
#         """ Local evolution exp(iθ(YX-XY)) *maybe θ/2..? """
#         Ω = qiskit.QuantumCircuit(2)
#         Ω.sdg(0)
#         Ω.h(0)
#         Ω.h(1)
#         Ω.cx(1,0)
#         Ω.rz(θ,0)
#         Ω.cx(1,0)
#         Ω.sx(0)
#         Ω.sxdg(1)
#         Ω.cx(1,0)
#         Ω.rz(-θ,0)
#         Ω.cx(1,0)
#         Ω.h(0)
#         Ω.h(1)
#         Ω.s(1)
#         return Ω.to_gate()
#
#     Hk = H.Hk(k)
#
#     oddswap = np.arange(H.M)
#     evnswap = np.arange(H.M)
#     for q in range(H.M-1):
#         if q & 1:   oddswap[[q,q+1]] = oddswap[[q+1,q]]
#         else:       evnswap[[q,q+1]] = evnswap[[q+1,q]]
#
#     index = np.arange(H.M)
#     U = qiskit.QuantumCircuit(H.M)
#     for q in range(H.M):
#         for qq in range(q&1, H.M-1, 2):
#             α, β = index[qq], index[qq+1]
#             θr = Hk[α,β].real/2 * coef
#             θi = Hk[α,β].imag/2 * coef
#
#             U.append(xxyy(θr),[qq,qq+1])
#             # U.barrier([qq,qq+1])
#             U.append(yxxy(θi),[qq,qq+1])
#             U.swap(qq, qq+1)
#         # U.barrier()
#         if q & 1:   index = index[oddswap]
#         else:       index = index[evnswap]
#
#     for q in range(H.M):
#         α = index[q]
#         θ = -Hk[α,α].real * coef
#         U.rz(θ, q)
#     # U.barrier()
#
#     for q in reversed(range(H.M)):
#         if q & 1:   index = index[oddswap]
#         else:       index = index[evnswap]
#         for qq in range(q&1, H.M-1, 2):
#             α, β = index[qq], index[qq+1]
#             θr = Hk[α,β].real/2 * coef
#             θi = Hk[α,β].imag/2 * coef
#
#             U.swap(qq, qq+1)
#             U.append(yxxy(θi),[qq,qq+1])
#             # U.barrier([qq,qq+1])
#             U.append(xxyy(θr),[qq,qq+1])
#         # U.barrier()
#
#     return U

def iqpenergy(operator, circuit, iter=10):
    """ Use IQPE algorithm to find nearest eigenvalue to point.

    Parameters
    ----------
    operator: qiskit.aqua.BaseOperator
    circuit: qiskit.QuantumCircuit
    iter: int
        number of iterations to recur QPE; increases precision

    Returns
    -------
    E: float
        energy estimated by IQPE
    """
    n = circuit.num_qubits
    state_in = qiskit.aqua.components.initial_states.Custom(n, circuit=circuit)
    return qiskit.aqua.algorithms.IQPE(
        operator=operator,
        state_in=state_in,
        num_iterations=iter,
        expansion_mode='suzuki',
        expansion_order=2,
        quantum_instance=qiskit.aqua.QuantumInstance(
            qiskit.Aer.get_backend('qasm_simulator')
        ),
    ).compute_minimum_eigenvalue()['eigenvalue'].real


def eigexpand(H, point):
    """ Expand experimental eigenstates into analytic eigenstate decomposition.

    Parameters
    ----------
    H: Hamiltonian
    point: dict
        element of `points` list in VQD object

    Returns
    -------
    lap: ndarray
        lap[i,j] gives projection of varied state i on true eigenstate j
    """
    # CALCULATE ANALYTIC EIGENSTATES
    kpt = np.array(point['ka'])
    E0, U0 = np.linalg.eigh(H.Hk(kpt))

    # CONVERT TO HILBERT SPACE, SO u0[l,:] GIVES QUBIT STATEVECTOR
    u0 = np.zeros((H.M,2**H.M), dtype=complex)
    for l in range(H.M):
        for i in range(H.M):
            u0[l,2**i] = U0[i,l]

    # FOR EACH LEVEL, CONVERT PARAMETERS TO STATEVECTOR AND PROJECT
    lap = np.empty((H.M,H.M), dtype=complex)
    ansatz = qctb.circuit.ansatz(H.M)
    for level in point['levels']:
        i = level['l']
        x = level['x']

        # VARIED EIGENSTATE
        u = qiskit.execute(
            qctb.circuit.bind(ansatz, x),
            qiskit.Aer.get_backend('statevector_simulator'),
        ).result().get_statevector()

        # PROJECT ONTO TRUE EIGENSTATES
        for j in range(H.M):
            lap[i,j] = np.vdot(u, u0[j])

    return lap
