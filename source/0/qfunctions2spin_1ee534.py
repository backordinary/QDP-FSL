# https://github.com/ZZSquare/InteractionRepresentation/blob/17eb05fba3efe0197da33f06cf419689db8a66ea/qfunctions2spin.py
import numpy as np
from scipy.linalg import expm
from scipy.interpolate import Akima1DInterpolator
from qiskit import QuantumCircuit

X = np.matrix([[0, 1.0], [1.0, 0]], dtype='complex128')
Y = np.matrix([[0, -1.0j], [1.0j, 0]], dtype='complex128')
Z = np.matrix([[1.0, 0], [0, -1.0]], dtype='complex128')
I = np.matrix([[1.0, 0], [0, 1.0]], dtype='complex128')
XY_MATRIX = np.kron(X, X) + np.kron(Y, Y)
BX_MATRIX = np.kron(I, X) + np.kron(X, I)
BY_MATRIX = np.kron(I, Y) + np.kron(Y, I)
BZ_MATRIX = np.kron(I, Z) + np.kron(Z, I)

def XYMatrix(J):
    return J * (-0.25) * XY_MATRIX

def BxMatrix(Bx):
    return Bx * BX_MATRIX

def ByMatrix(By):
    return By * BY_MATRIX

def BzMatrix(Bz):
    return Bz * BZ_MATRIX


def delta(XY, B_x, Bz):
    B_z = BzMatrix(Bz)
    eigenvalues, _ = np.linalg.eigh(XY + B_x + B_z)
    ordereigenvalues = np.sort(eigenvalues)
    return ordereigenvalues[0] - ordereigenvalues[1]


def adiabaticramp(J, Bx, dB, Bz_max, dt_steps, dt_steps_bool, g):
    XY = XYMatrix(J)
    B_x = BxMatrix(Bx)

    intervals = int(Bz_max / dB)
    Bz = np.zeros(intervals + 1)
    Bz[0] = Bz_max
    time = np.zeros(intervals + 1)
    time[0] = 0
    thisTime = 0

    for i in range(0, intervals):
        thisBz = Bz_max - dB * (i + 1)
        Bz[i + 1] = thisBz
        dt = g * dB / delta(XY, B_x, thisBz) ** 2
        thisTime += dt
        time[i + 1] = thisTime

    print('Adiabatic Ramp Time: ', str(time[-1]))
    if dt_steps_bool == 'dt':
        max_t = time[-1] - time[-1] % dt_steps
        time_steps_dt = max_t / dt_steps
        print(time_steps_dt)
        time_grid = np.linspace(0, max_t, int(time_steps_dt))
        dt = dt_steps
    else:
        time_grid = np.linspace(0, time[-1], dt_steps + 1)
        time_steps_dt = time_grid[1] - time_grid[0]
        dt = time_steps_dt

    f = Akima1DInterpolator(time, Bz)
    Bz_grid = f(time_grid)

    return time_grid, Bz_grid, time_steps_dt, dt

def XYUnitary(XY, B_x, Bz, dt):
    m = XY + B_x + BzMatrix(Bz)
    return expm(-1.j*m*dt)

def exactm(J, Bx, Bzarray, dt):
    U = np.kron(I, I)
    XY = XYMatrix(J)
    B_x = BxMatrix(Bx)
    psi0 = np.matrix([[0], [0], [0], [1]])
    mzarray = []

    for bz in Bzarray:
        U = np.matmul(XYUnitary(XY, B_x, bz, dt), U)
        psi = np.matmul(U, psi0)
        m = np.matmul(psi.getH(), np.matmul(BzMatrix(1), psi))
        mzarray.append(-np.real(m.item(0)) / 2)
    return mzarray


def Hxy_gate(J, dt):
    Hxy = QuantumCircuit(2, name='exp{-i*Hxy*dt}')

    # w2 = conjugate_transpose{w1}
    Hxy.rx(-np.pi / 2, 0)
    Hxy.rx(-np.pi / 2, 1)

    # cnot
    Hxy.cx(0, 1)

    # u2 = {{cos(theta), -isin(theta)}, {-isin(theta), cos(theta)}}
    Hxy.rx(2 * -0.25 * J * dt, 0)
    # v2 = {{exp{-it}, 0}, {0, exp{it}}
    Hxy.rz(2 * -0.25 * J * dt, 1)

    # cnot
    Hxy.cx(0, 1)

    # w1
    Hxy.rx(np.pi / 2, 0)
    Hxy.rx(np.pi / 2, 1)

    return Hxy.to_instruction()


def Bx_gate(Bx, theta, dt):
    circ = QuantumCircuit(2, name='exp{-i*Bx*dt}')

    # u2 = {{cos(theta), -isin(theta)}, {-isin(theta), cos(theta)}}
    circ.rx(2 * Bx * np.cos(theta) * dt, 0)
    # Identity
    circ.i(1)
    circ.i(0)
    # u2 = {{cos(theta), -isin(theta)}, {-isin(theta), cos(theta)}}
    circ.rx(2 * Bx * np.cos(theta) * dt, 1)

    return circ.to_instruction()


def By_gate(Bx, theta, dt):
    circ = QuantumCircuit(2, name='exp{-i*By*dt}')

    circ.ry(2 * Bx * np.sin(theta) * dt, 0)
    # Identity
    circ.i(1)
    circ.i(0)

    circ.ry(2 * Bx * np.sin(theta) * dt, 1)

    return circ.to_instruction()


def Bxy_gate(Bx, theta, dt):
    circ = QuantumCircuit(2, name='exp{-i*Bxy*dt}')

    # circ.rx(2 * Bx * np.cos(theta) * dt, 0)
    # circ.ry(2 * Bx * np.sin(theta) * dt, 0)
    circ.u(2 * Bx * dt, theta + np.pi / 2, -theta - np.pi / 2, 0)

    circ.i(1)
    circ.i(0)

    # circ.rx(2 * Bx * np.cos(theta) * dt, 1)
    # circ.ry(2 * Bx * np.sin(theta) * dt, 1)
    circ.u(2 * Bx * dt, theta + np.pi / 2, -theta - np.pi / 2, 1)

    return circ.to_instruction()


def HxyByBx_gate(J, Bx, theta, dt):
    HxyByBx = QuantumCircuit(2, name='HxyByBx')

    HxyByBx.append(Hxy_gate(J, dt), [0, 1])
    HxyByBx.append(Bxy_gate(Bx, theta, dt), [0, 1])

    return HxyByBx.to_instruction()


def twospin_instruction(circ, J, Bx, theta, dt):
    for t in theta:
        circ.append(HxyByBx_gate(J, Bx, t, dt), [0, 1])


def BxyUnitary(Bx, theta, dt):
    return expm(-1.j * dt * (BxMatrix(Bx * np.cos(theta)) + ByMatrix(Bx * np.sin(theta))))

def interactionUnitary(XY, Bx, theta, dt):
    return np.matmul(XY, BxyUnitary(Bx, theta, dt))

def XYUnitary0(J, dt):
    return expm(-1.j * XYMatrix(J) * dt)

def interactionm(J, Bx, Theta, dt):
    U = np.kron(I, I)
    XY = XYUnitary0(J, dt)
    psi0 = np.matrix([[0], [0], [0], [1]])
    mzarray = []

    for theta in Theta:
        U = np.matmul(interactionUnitary(XY, Bx, theta, dt), U)
        psi = np.matmul(U, psi0)
        m = np.matmul(psi.getH(), np.matmul(BzMatrix(1), psi))
        mzarray.append(-np.real(m.item(0)) / 2)
    return mzarray


def theta(Bzarray, dt):
    theta = 0
    thetaarray = []
    time = []
    t = 0
    for bz in Bzarray:
        theta += bz * dt
        t += dt
        time.append(t)
        thetaarray.append(-2 * theta)

    return thetaarray, dt