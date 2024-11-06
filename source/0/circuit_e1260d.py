# https://github.com/PVNkT/HHL/blob/29c91c32d4e251e99aeb7d7b43ce27b197d7e7e1/circuit.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from QPE import QPE, qpe_qiskit
from rotation import rotation, Reciprocal, my_rotation
from initialize import make_b_state
def circuit(A, b, nl, evolution_time, delta, neg_vals, wrap = True, measurement = None):
    #flag qubit의 갯수, 상태 준비가 확률적인 경우 늘릴 필요가 있음
    nf = 1
    #evolution time 정의, eigenvalue에 따라서 다르게 정의할 필요가 있음
    t = evolution_time
    #eigenvalue가 음수일 경우 -를 포함하는 qubit이 필요하고 이것을 boolean으로 표현
    neg_vals = neg_vals
    #b벡터를 normalize된 양자 상태로 바꾸는 회로
    init_b, nb = make_b_state(b)
    #Quantum Phase Estimation을 시행하는 회로
    qc_qpe= QPE(nl, nb, A, t)
    #QPE를 역과정으로 만든 회로
    qc_qpet = QPE(nl, nb, A, t).inverse()
    """
    #qiskit에서 제공하는 QPE 함수를 그대로 사용하는 경우 아래의 코드를 사용
    qc_qpe, nb = qpe_qiskit(nl, A, t)
    qc_qpet = qpe_qiskit(nl, A, t, adjoint = False)[0].inverse()
    """
    #양자 register들을 설정, nl(QPE에 필요한 상태 qubit의 수), nb(b벡터를 표현하는 양자 register), nf(reciprocal 회전을 적용하는 qubit 일반적으로 1), na(필요할 경우 사용, ancila qubit), 
    nl_rg = QuantumRegister(nl, "state")
    nb_rg = QuantumRegister(nb, "q")
    #na_rg = QuantumRegister(nl, "a")
    nf_rg = QuantumRegister(nf, "flag")
    #classical register들을 설정, 계산에 필요한 f와 b만을 따로 측정
    if measurement == "norm":
        qc = QuantumCircuit(nb_rg, nl_rg,nf_rg)
    else: 
        cf = ClassicalRegister(nf)
        cb = ClassicalRegister(nb)
        #register들을 합쳐서 회로를 구성, 필요할 경우 na추가
        qc = QuantumCircuit(nb_rg, nl_rg,nf_rg, cf, cb)
    qc.barrier()
    """
    #Chebyshev근사를 사용한 방식을 사용하는 경우 (부정확함)
    qc_rot = rotation(nl)
    """
    #qiskit에서 제공하는 Reciprocal 함수를 사용해서 f qubit에 대한 회전을 구현 
    #qc_rot = Reciprocal(nl, delta = delta, neg_vals = neg_vals)
    qc_rot = my_rotation(nl, nf=nf, scaling = delta, neg_vals = neg_vals)
    if wrap:
        #QPE, reciprocal, QPE inverse를 순서대로 추가, 각각을 하나의 instruction으로 표현
        qc.append(init_b, nb_rg[:])
        qc.barrier()
        qc.append(qc_qpe,nl_rg[:]+nb_rg[:])
        qc.append(qc_rot,[nl_rg[i] for i in reversed(range(nl))]+nf_rg[:])
        qc.append(qc_qpet,nl_rg[:]+nb_rg[:])
    else:
        #QPE, reciprocal, QPE inverse를 순서대로 추가, 내부의 gate들을 나누어서 표현
        qc = qc.compose(init_b, nb_rg[:])
        qc.barrier()
        qc = qc.compose(qc_qpe,nl_rg[:]+nb_rg[:])
        qc = qc.compose(qc_rot,[nl_rg[i] for i in reversed(range(nl))]+nf_rg[:])
        qc = qc.compose(qc_qpet,nl_rg[:]+nb_rg[:])
    qc.barrier()
    #각 register들을 측정한다
    if measurement == "statevector" or measurement == "norm":
        pass
    else:
        qc.measure(nf_rg,cf)
        qc.barrier()
        qc.measure(nb_rg,cb)
    #회로 그림을 저장
    if wrap:
        qc.draw("mpl").savefig("image/HHL_circuit_wrapped.png")
    else:
        qc.draw("mpl").savefig("image/HHL_circuit_unwrapped.png")
    #회로를 반환
    return qc