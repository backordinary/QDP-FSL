# https://github.com/thiagom123/Circuito-Quantico-Parametrizado/blob/0866c7cf39a42ff5d446b8c02f7f943563e530b0/CircuitoParametrizado.py
import numpy as np
import matplotlib
import qiskit
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit, ClassicalRegister
from gates import operador_controlado, Z
from CircuitoEncoder import CircuitoEncoder
from qiskit.tools.visualization import circuit_drawer

def CQV(params, X):
    '''

    :param params: Vetor dos parâmetros
    :X: Vetor de características
    :return: retorna o ciruito parametrizado e após o encoding
    '''
    #n = np.size(X)
    qr = QuantumRegister(1, name="q")
    cr = qiskit.ClassicalRegister(1, name="c")
    qc = QuantumCircuit(qr, cr)
    qc.append(CircuitoEncoder([X], 1), qargs=[qr], cargs=[cr])
    #print(params)
    qc.u3(theta=params[0], phi=params[1], lam=params[2], qubit=qr)
    qc.measure(qr, cr[0])
    return qc


def TesteHadamardBásico(Theta):
    '''
    :param n: Dimensão do vetor de características
    :param Theta: Vetor de Parâmetros do modelo de ML
    :return: derivada de M em relação a Theta
    '''
    q1 = QuantumRegister(1, name='q')
    anc = QuantumRegister(1, name='ancilla')
    cl = ClassicalRegister(1, name='cl')
    circuito1: QuantumCircuit = QuantumCircuit(q1, anc, cl)
    # Porta de Hadamard na Ancilla
    circuito1.h(anc)
    # Unitária no Sistema
    circuito1.p(Theta, 0)
    # Operador Controlado no sistema
    circuito1.append(operador_controlado((-1) * 1j * Z()), [q1, anc])
    # Medição Controlada no sistema
    circuito1.append(operador_controlado(Z()), [q1, anc])
    circuito1.z(anc)
    circuito1.measure(anc, cl)
    return circuito1

def TesteHadamard(n, Theta, j, t):
    '''
    :param n: Dimensão do vetor de características
    :param Theta: Vetor de Parâmetros do modelo de ML
    :param j: Indice do Theta_j da Derivada
    :param t: Tamanho do vetor Theta
    :return: circuito pronto para calcular a derivada do valor esperado de M
    '''
    q1 = QuantumRegister(n, name='q')
    anc = QuantumRegister(1, name='ancilla')
    cl = ClassicalRegister(n, name='cl')
    circuito1: QuantumCircuit = QuantumCircuit(q1, anc, cl)
    # Porta de Hadamard na Ancilla
    circuito1.h(anc)
    # Unitária no Sistema
    for k in range(j):
        for i in range(n):
            circuito1.p(Theta[k], i)
    # Operador Controlado no sistema
    circuito1.append(operador_controlado((-1) * 1j * Z()), [q1, anc])
    # Unitária no Sistema
    for k in range(j, t):
        for i in range(n):
            circuito1.p(Theta[k], i)
    # Medição Controlada no sistema
    circuito1.append(operador_controlado(Z()), [q1, anc])
    circuito1.z(anc)
    circuito1.measure(anc, cl)
    return circuito1




if __name__ == '__main__':
    '''
    Nesse main nós testamos a função do TesteHadamardBásico
    '''
    # circuito = CircuitoParametrizado([1], 1, np.pi, 1)
    circuito = TesteHadamardBásico(1, np.pi)
    print("Teste")
    print(circuito.draw())
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuito, backend)
    result = job.result()
    print("Counts Z=1")
    print(result.get_counts()['0'])
    print("Counts Z=-1")
    print(result.get_counts()['1'])
    circuito = TesteHadamardBásico(1, np.pi/2)
    print("Teste")
    print(circuito.draw())
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuito, backend)
    result = job.result()
    print("Counts Z=1")
    print(result.get_counts()['0'])
    print("Counts Z=-1")
    print(result.get_counts()['1'])
