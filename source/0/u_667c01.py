# https://github.com/EmilBahnsenMigration/masters-source/blob/a8f7626e6215afbebf73ca294f89c71eacd93808/diamondQiskit/U.py
from qiskit import *
import math

π = math.pi


# --- Construct U ---
# omega = t_g · J_C
def U(omega):
    # U_A
    UA = QuantumCircuit(2, name='U_A')
    UA.cx(1,0)
    UA.ch(0,1)
    UA.cx(1,0)

    # U_B
    UB = QuantumCircuit(2, name='U_B')
    UB.cz(0,1)
    UB.swap(0,1)
    UB.z(0)
    UB.z(1)

    # U_C
    UC = QuantumCircuit(3, name='U_C')
    UC.h(2)
    UC.ccx(0,1,2)
    UC.h(2)
    UC.cswap(0,1,2)
    UC.cz(0,1)
    UC.cz(0,2)
    UC.rz(omega,0)

    # U_D
    UD = QuantumCircuit(3, name='U_D')
    UD.h(2)
    UD.ccx(0,1,2)
    UD.h(2)
    UD.cswap(0,1,2)
    UD.z(0)
    UD.rz(-omega,0)

    cr = QuantumRegister(2, name='c')
    tr = QuantumRegister(2, name='t')

    U = QuantumCircuit(cr, tr)
    if π % omega == 0:
        U.name = 'U(π/{})'.format(round(π/omega))
    elif omega % π == 0:
        U.name = 'U({}π)'.format(round(omega/π))
    else:
        U.name = 'U({})'.format(omega)

    U.append(UA, [0,1])
    U.append(UB, [2,3])
    U.append(UC, [1,2,3])
    U.append(UD, [0,2,3])
    U.append(UA, [0,1])

    return U
