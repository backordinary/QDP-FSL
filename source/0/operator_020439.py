# https://github.com/marcobn/Quantum/blob/0b12d7ce39d589e0810c18f7b0eb183120c6ce38/BandTheory/qctb/operator.py
""" Manually construct operator for periodic system minimizing measurements. """

import itertools

import numpy as np
import qiskit
import qctb

from qiskit.aqua.operators import I, X, Y, Z
from qiskit.aqua.operators import SummedOp, TensoredOp, ComposedOp
from qiskit.aqua.operators import StateFn, CircuitStateFn

def ReciprocalOperator(H,k,ansatz):
    """ Measurement op composing Hamiltonian and ansatz, reciprocal approach.

    Parameters
    ----------
    H: tightbinding.Hamiltonian
        the periodic system to measure
    k: list of int with length H.D
        kpt of interest
    ansatz: qiskit.QuantumCircuit
        circuit preparing orbitals of the principal unit cell

    Returns
    -------
    energyop: qiskit.aqua.operators.SummedOp
        grouped measurement operator evaluating energy of periodic system

    """
    Hk = H.Hk(k)

    groups = []

    # PREPARE Z GROUP
    oplist = []
    E0 = np.sum(np.diag(Hk))                                # SHIFT ENERGY
    if E0 != 0:
        oplist.append( np.sum(np.diag(Hk)) * word(H.M) )
    for α in range(H.M):
        if Hk[α,α] == 0: continue
        oplist.append( -Hk[α,α] * word(H.M, [α]) )          # ONSITE ENERGY
    if oplist:
        op = SummedOp(oplist, coeff=0.5, abelian=True)
        groups.append( measure(op, ansatz, 'Z'*H.M) )

    # PREPARE X AND Y GROUPS
    oplist = []
    for α in range(H.M-1):
        for β in range(α+1,H.M):
            if Hk[α,β].real == 0: continue
            oplist.append( Hk[α,β].real * word(H.M, [α,β]) )
    if oplist:
        op = SummedOp(oplist, coeff=0.5, abelian=True)
        groups.append( measure( op, ansatz, 'X'*H.M ) )
        groups.append( measure( op, ansatz, 'Y'*H.M ) )

    # PREPARE XY GROUPS
    basisXY = 'X' + 'Y'*(H.M-1)
    basisYX = 'Y' + 'X'*(H.M-1)
    for α in range(H.M-1):
        oplistYX = []
        oplistXY = []
        for β in range(α+1,H.M):
            if Hk[α,β].imag == 0: continue
            oplistYX.append(  Hk[α,β].imag * word(H.M, [α,β]) )
            oplistXY.append( -Hk[α,β].imag * word(H.M, [α,β]) )
        if oplistXY:
            op = SummedOp(oplistYX, coeff=0.5, abelian=True)
            groups.append( measure( op, ansatz, basisYX ) )
            op = SummedOp(oplistXY, coeff=0.5, abelian=True)
            groups.append( measure( op, ansatz, basisXY ) )
        basisXY = basisXY[:α] + 'IX' + basisXY[α+2:]
        basisYX = basisYX[:α] + 'IY' + basisYX[α+2:]

    return SummedOp(groups)




def RealspaceOperator(H,n,ansatz):
    """ Measurement op composing Hamiltonian and ansatz, real-space approach.

    Parameters
    ----------
    H: tightbinding.Hamiltonian
        the periodic system to measure
    n: list of int with length H.D
        number of qubits in each site register
    ansatz: qiskit.QuantumCircuit
        circuit preparing orbitals of the principal unit cell

    Returns
    -------
    energyop: qiskit.aqua.operators.SummedOp
        grouped measurement operator evaluating energy of periodic system

    """

    # CONVERT HOPPING PARAMETERS INTO MORE CONVENIENT FORM
    Hx = {}     # MAPS Δ : (α,β)
    for β, hops in enumerate(H.t):
        for hop in hops:
            t  = hops[hop]              # THE ACTUAL HOPPING PARAMETER
            α  = hop[0]                 # THE TARGET ORBITAL
            Δ  = tuple(hop[1:])         # THE TARGET UNIT CELL
            nΔ = tuple(-δ for δ in Δ)   # NEGATION OF TARGET UNIT CELL

            if not (Δ in Hx or nΔ in Hx):
                Hx[Δ] = np.zeros((H.M,H.M)) # INITIALIZE MATRIX
            if Δ in Hx:
                Hx[Δ][α,β] = t              # ASSIGN HOPPING PARAMETER
            # if nΔ in Hx, trust t to be added when nΔ is Δ (ie. skip)

    # REAL AND IMAGINARY PARTS ON ORBITAL REGISTER
    AM = {}
    BM = {}

    for Δ in Hx:
        # REAL PART
        AM[Δ] = []

        # PREPARE Z GROUP
        oplist = []
        E0 = np.sum(np.diag(Hx[Δ]))                         # SHIFT ENERGY
        if E0 != 0:
            oplist.append( E0 * word(H.M) )
        for α in range(H.M):
            if Hx[Δ][α,α] == 0: continue
            oplist.append( -Hx[Δ][α,α] * word(H.M, [α]) )   # ONSITE ENERGY
        if oplist:
            op = SummedOp(oplist, coeff=0.5, abelian=True)
            AM[Δ].append( measure(op, ansatz, 'Z'*H.M) )

        # PREPARE X AND Y GROUPS
        oplist = []
        for α in range(H.M-1):
            for β in range(α+1,H.M):
                if Hx[Δ][α,β]+Hx[Δ][β,α] == 0: continue
                oplist.append( (Hx[Δ][β,α]+Hx[Δ][α,β]) * word(H.M, [α,β]) )
        if oplist:
            op = SummedOp(oplist, coeff=0.25, abelian=True)
            AM[Δ].append( measure( op, ansatz, 'X'*H.M ) )
            AM[Δ].append( measure( op, ansatz, 'Y'*H.M ) )

        AM[Δ] = SummedOp(AM[Δ])

        # IMAGINARY PART
        BM[Δ] = []

        # PREPARE XY GROUPS
        basisXY = 'X' + 'Y'*(H.M-1)
        basisYX = 'Y' + 'X'*(H.M-1)
        for α in range(H.M-1):
            oplistXY = []
            oplistYX = []
            for β in range(α+1,H.M):
                if Hx[Δ][β,α]-Hx[Δ][α,β] == 0: continue
                oplistYX.append(  (Hx[Δ][β,α]-Hx[Δ][α,β]) * word(H.M, [α,β]) )
                oplistXY.append( -(Hx[Δ][β,α]-Hx[Δ][α,β]) * word(H.M, [α,β]) )
                # if Hx[Δ][α,β].imag == 0: continue
                # oplist.append( Hx[Δ][α,β].imag * word(H.M, [α,β]) )
            if oplistXY:
                op = SummedOp(oplistYX, coeff=0.25, abelian=True)
                BM[Δ].append( measure( op, ansatz, basisYX ) )
                op = SummedOp(oplistXY, coeff=0.25, abelian=True)
                BM[Δ].append( measure( op, ansatz, basisXY ) )
            basisXY = basisXY[:α] + 'IX' + basisXY[α+2:]
            basisYX = basisYX[:α] + 'IY' + basisYX[α+2:]

        BM[Δ] = SummedOp(BM[Δ])

    # REAL AND IMAGINARY PARTS OF |ν+1⟩⟨ν| (FOR EACH SITE REGISTER)
    AN = {}
    BN = {}
    for i in range(H.D):
        AN[i], BN[i] = SiteOperators(n[i], i)

    # MERGE REGISTERS
    groups = []
    for Δ in Hx:
        for z in range(2**(1+H.D)):
            ops = []
            phase = 1

            # PICK WHICH ORBITAL OPERATION
            if z & 1:                       # IMAGINARY CONTRIBUTION
                ops.append(BM[Δ])
                phase *= 1j
            else:                           # REAL CONTRIBUTION
                ops.append(AM[Δ])

            # PICK EACH SITE OPERATION
            for i in range(H.D):
                z >>= 1
                if z & 1:                   # IMAGINARY CONTRIBUTION
                    if Δ[i] == 0:               # HANDLE |ν⟩⟨ν| on z==0
                        phase = 1j  # CHEAP TRICK TO BREAK OUT OF OUTER LOOP
                        break       #   WITHOUT ADDING OPS
                    ops.append(BN[i])
                    phase *= 1j * Δ[i]
                else:                       # REAL CONTRIBUTION
                    if Δ[i] == 0:               # |ν⟩⟨ν| CONTRIBUTES JUST I
                        # ops.append(nullop(n[i]))
                        ops.append(NullOperator(n[i]))
                    else:
                        ops.append(AN[i])

            # CONSIDER ONLY REAL TERMS
            if phase.imag: continue
            # NOTE: Alternatively we can get phase directly from z and Δ,
            #       and skip before selecting ops.

            # DOUBLE REAL PART, EXCEPT INTRACELL Δ=(0..0)
            if any(Δ):  phase *= 2

            # MERGE SELECTED OPERATIONS AND APPLY PHASE
            mergedops = merge(*ops)
            mergedops = [mergedop * phase for mergedop in mergedops]
            groups.extend(mergedops)


    return SummedOp(groups)



def XYOperator(coeffs, parity, ansatz):
    """ Create measurement operator on commuting group of XY words.

    Parameters
    ----------
    coeffs: ndarray
        coefficients for each XY word
        coeffs[i] corresponds to word with i's binary representation,
            prepending a parity bit, and converting 0->X, 1->Y.
            Pad on the left with I's to match ansatz size.
    parity: int
        0 if commuting strings have even number of Y, or 1 if odd
    ansatz: qiskit.QuantumCircuit
        operation to apply before measurement

    Returns
    -------
    op: qiskit.aqua.operators.ComposedOp
        measurement operator which simultaneously measures all words at once

    """
    # GET LENGTH OF EACH STRING
    n = ansatz.num_qubits
    r = int(np.log2(len(coeffs)))

    # PREFIX FOR EVERY TERM
    prefix = [I for q in range(n-1-r)] + [Z]    # IDENTITY SPACE AND PARITY BIT

    # CREATE ABELIAN SUM
    oplist = []
    for i, coeff in enumerate(coeffs):
        if not coeff: continue                      # SKIP ZERO TERMS
        ibin = np.binary_repr(i,r) if r else ""     # i AS BINARY STRING
        wt = ibin.count("1")                        # WEIGHT OF STRING
        letters = prefix + [Z if ib=="1" else I for ib in ibin]
        letters = reversed(letters)     # FLIP ORDER SO OPERATOR eval() WORKS
        pauli = TensoredOp(letters).to_pauli_op()
        phase = (-1)**( (1-parity)*(((wt&2)>>1)^(wt&1)) + parity*((wt&2)>>1) )

        oplist.append( phase * coeff * pauli )
    op = SummedOp(oplist, abelian=True)

    # APPEND MULTI-QUBIT BASIS ROTATION TO ANSATZ
    ct = ansatz.compose(
        qctb.circuit.measure_xy(r+1, parity),
        qubits=range( n-1 - r, n ),
    )

    basis = 'I'*(n-r-1) + 'Z' + 'X'*r
    return measure(op, ct, basis=basis)

def SiteOperators(n, ni):
    """ Construct real and imaginary parts of operator ∑|ν+1⟩⟨ν|

    Parameters
    ----------
    n: int
        number of qubits in this site register
    ni: int
        dimension index (required for distinguishing parameters)

    Returns
    -------
    A: qiskit.aqua.operators.SummedOp
        grouped measurement operators for real part of ∑|ν+1⟩⟨ν|
    B: qiskit.aqua.operators.SummedOp
        grouped measurement operators for imaginary part of ∑|ν+1⟩⟨ν|

    """
    # GENERATE CIRCUIT
    ansatz = qctb.circuit.sitecircuit(n,ni)

    oplist_A = []
    oplist_B = []

    for r in range(n):

        # INITIALIZE COEFFICIENTS FOR REAL AND IMAGINARY PARTS
        a = np.ones(2**r) / 2**(r+1)
        b = np.ones(2**r) / 2**(r+1)

        # ADJUST PHASES
        for i in range(2**r):
            ibin = np.binary_repr(i,r)                  # i AS BINARY STRING
            wt = ibin.count("1")                        # WEIGHT OF STRING
            k = (wt >> 1) & 1                           # PARITY OF PAIRS
            a[i] *= (-1)**( k )
            b[i] *= (-1)**( k ^ (1-(wt&1)) )

        # ADD |0..0⟩⟨1..1|
        if r == n-1:
            # FORTUITOUSLY, COEFFICIENTS WORK OUT THIS WAY
            a, b = a-b, a+b

        # ADD OPERATORS
        if any(a): oplist_A.append( XYOperator( a, 0, ansatz ) )
        if any(b): oplist_B.append( XYOperator( b, 1, ansatz ) )

    # RETURN REAL AND IMAGINARY OPERATORS
    return SummedOp(oplist_A), SummedOp(oplist_B)

def OverlapOperator(ct0, ct1):
    """ Overlap measurement operator between two circuits.

    Parameters
    ----------
    ct0, ct1: qiskit.QuantumCircuit
        the two circuits to be compared

    Returns
    -------
    overlap_op: qiskit.aqua.operators.SummedOp
        measurement operator composing projection |0⟩⟨0| with circuit ct1† ct0
    """
    n = ct0.num_qubits

    # AbelianSummedOp OF IZ COMBINATIONS
    oplist = []
    for i in range(2**n):
        ibin = np.binary_repr(i,n)
        zlist = np.nonzero([int(ib) for ib in ibin])[0]
        oplist.append( word(n, zlist ) )
    op = SummedOp(oplist, coeff=2**-n, abelian=True)

    # OVERLAP CIRCUIT
    ct = ct0 + ct1.inverse()

    # WRAP INTO SINGLE-ELEMENT SummedOp
    return SummedOp( [measure(op, ct)] )

def NullOperator(n):
    """ Identity measurement operator.

    Parameters
    ----------
    n: int
        number of qubits to act on

    Returns
    -------
    null_op:
        identity measurement operator on n qubits

    """

    # AbelianSummedOp OF...JUST I...
    op = SummedOp([word(n)], abelian=True)

    # IDENTITY CIRCUIT
    ct = qiskit.QuantumCircuit(n)
    for q in range(n):
        ct.i(q)

    # WRAP INTO SINGLE-ELEMENT SummedOp
    return SummedOp( [measure(op, ct)] )



def merge(*ops):
    """ Merge operators acting on different spaces into a single operator.

    Parameters
    ----------
    *ops: qiskit.aqua.operators.SummedOp
        grouped measurement operators to merge

    Returns
    -------
    mereged_op: qiskit.aqua.operators.SummedOp
        grouped measurement operator with tensored operators and circuits

    """
    mergedops = []
    for brakets in itertools.product(*ops):
        # brakets IS A TUPLE OF StateFns

        # braket[0] WRAPS THE ABELIAN SUMMED OP ON braket's REGISTER
        groups = [braket[0].primitive for braket in brakets]
        mergedgroup = np.prod([group.coeff for group in groups]) * SummedOp([
            TensoredOp(list(terms[::-1])).to_pauli_op() # REVERSE ORDER
            for terms in itertools.product(*groups)
        ], abelian=True)

        # braket[1] WRAPS THE CIRCUIT ON braket's REGISTER
        cts = [braket[1].primitive for braket in brakets]
        n = [ct.num_qubits for ct in cts]
        mergedct = qiskit.QuantumCircuit(sum(n))
        qI = 0                  # QUBIT INDEX TO START FROM
        for i in range(len(n)):
            mergedct.compose(cts[i], qubits=range(qI,qI+n[i]), inplace=True)
            qI = qI + n[i]      # UDATE QUBIT INDEX TO START FROM

        mergedops.append( measure(mergedgroup, mergedct) )
    return mergedops




def word(n, zlist=[]):
    """ Create IZ word as a PauliOperator.

    Parameters
    ----------
    n: int
        length of word
    zlist: list of int
        indices which contain Z as opposed to I

    Returns
    -------
    pauli_op: qiskit.aqua.operators.PauliOp

    """
    letters = []
    for i in range(n):
        if i in zlist:  letters.append(Z)
        else:           letters.append(I)
    letters = letters[::-1]         # FLIP ORDER SO OPERATOR eval() WORKS
    return TensoredOp(letters).to_pauli_op()

def measure(op, ansatz, basis=""):
    """ Combine operator and circuit into a measurement operator.

    Parameters
    ----------
    op: qiskit.aqua.operators.SummedOp
        abelian sum of IZ words
    ansatz: qiskit.QuantumCircuit
        circuit to prepare state
    basis: str
        Pauli label determining basis to measure in

    Returns
    -------
    measure_op: qiskit.aqua.operators.ComposedOp
        measurement operator with rotation gates appended to match basis

    """
    # ADD BASIS ROTATIONS
    ct = ansatz.copy()
    for q, σ in enumerate(basis):
        if σ == "X":
            ct.h(q)
        if σ == "Y":
            ct.z(q)
            ct.s(q)
            ct.h(q)

    # MERGE INTO STATE FUNCTIONS
    return StateFn(op, is_measurement=True) @ CircuitStateFn(ct)


def maxdepth(op):
    """ Calculate maximum depth of all circuits in op.

    Parameters
    ----------
    op: qiskit.aqua.operators.SummedOp or ComposedOp
        measurement operator, or sum thereof

    Returns
    -------
    depth: int
        longest depth of all circuits in each measurement group

    """
    # IF ONLY ONE MEASUREMENT GROUP, WRAP INTO SummedOp
    if isinstance(op, ComposedOp): op = SummedOp([op])
    # ACCESS ct AS braket[1].primitive
    return max( braket[1].primitive.depth() for braket in op )

def sumgates(op):
    """ Calculate number of gates in all circuits of op.

    Parameters
    ----------
    op: qiskit.aqua.operators.SummedOp or ComposedOp
        measurement operator, or sum thereof

    Returns
    -------
    gates: int
        total number of gates over all measurement groups

    """
    # IF ONLY ONE MEASUREMENT GROUP, WRAP INTO SummedOp
    if isinstance(op, ComposedOp): op = SummedOp([op])
    # ACCESS ct AS braket[1].primitive
    return sum( len(braket[1].primitive) for braket in op )

def maxgates(op):
    """ Calculate largest number of gates in a single circuit of op.

    Parameters
    ----------
    op: qiskit.aqua.operators.SummedOp or ComposedOp
        measurement operator, or sum thereof

    Returns
    -------
    gates: int
        largest gate count of all circuits in each measurement group

    """
    # IF ONLY ONE MEASUREMENT GROUP, WRAP INTO SummedOp
    if isinstance(op, ComposedOp): op = SummedOp([op])
    # ACCESS ct AS braket[1].primitive
    return max( len(braket[1].primitive) for braket in op )

def sumcxcnt(op):
    """ Calculate number of entanglements in all circuits of op.

    Parameters
    ----------
    op: qiskit.aqua.operators.SummedOp or ComposedOp
        measurement operator, or sum thereof

    Returns
    -------
    cxcnt: int
        total number of entanglements over all measurement groups

    """
    # IF ONLY ONE MEASUREMENT GROUP, WRAP INTO SummedOp
    if isinstance(op, ComposedOp): op = SummedOp([op])
    # ACCESS ct AS braket[1].primitive
    return sum( braket[1].primitive.count_ops().get('cx',0) for braket in op )

def maxcxcnt(op):
    """ Calculate largest number of entanglements in a single circuit of op.

    Parameters
    ----------
    op: qiskit.aqua.operators.SummedOp or ComposedOp
        measurement operator, or sum thereof

    Returns
    -------
    cxcnt: int
        largest entanglement count of all circuits in each measurement group

    """
    # IF ONLY ONE MEASUREMENT GROUP, WRAP INTO SummedOp
    if isinstance(op, ComposedOp): op = SummedOp([op])
    # ACCESS ct AS braket[1].primitive
    return max( braket[1].primitive.count_ops().get('cx',0) for braket in op )

def sumwords(op):
    """ Calculate number of pauli words in op.

    Parameters
    ----------
    op: qiskit.aqua.operators.SummedOp or ComposedOp
        measurement operator, or sum thereof

    Returns
    -------
    depth: int
        number of pauli words over all measurement groups

    """
    # IF ONLY ONE MEASUREMENT GROUP, WRAP INTO SummedOp
    if isinstance(op, ComposedOp): op = SummedOp([op])
    # ACCESS groups AS braket[0].primitive
    return sum( len(braket[0].primitive) for braket in op )

def sumabels(op):
    """ Calculate number of measurement groups in op.

    Parameters
    ----------
    op: qiskit.aqua.operators.SummedOp or ComposedOp
        measurement operator, or sum thereof

    Returns
    -------
    depth: int
        number of measurement groups

    """
    return 1 if isinstance(op, ComposedOp) else len(op)
