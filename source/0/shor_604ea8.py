# https://github.com/papparapa/quantum_computer/blob/efd6b95c4cd72f6090a3042a0a1743713bb9ff60/shor.py
# -*- coding: utf-8 -*-
import math
import numpy as np
from fractions import Fraction
from itertools import product
from qiskit import QuantumCircuit, QuantumProgram


# 与えられた数の二進表記を返すジェネレータ
def bin_digit(n):
    if n < 2:
        yield n
    else:
        for m in bin_digit(n // 2):
            yield m
    yield n % 2


# 二進表記の各桁のリストを整数型に変換
def inv_bin_digit(digit_list):
    num = 0
    for j, digit_j in enumerate(digit_list):
        num += 2**j * digit_j
    return num


# ユークリッドの互除法によりax+by=gcd(a,b)(0<a<b)の整数解をxが最小の自然数となるように求める
def solve_linear_indeterminate(a, b):
    r = [b, a]
    k = []
    while True:
        k.append(r[-2] // r[-1])
        r.append(r[-2] % r[-1])
        if r[-1] == 0:
            break
    solution = np.identity(2, dtype=int)
    for ki in k:
        solution = np.array([[0, 1], [1, -ki]], dtype=int) @ solution
    y, x = solution[0, 0], solution[0, 1]
    gcd = (solution @ np.array([b, a], dtype=int))[0]
    a_ = a // gcd
    b_ = b // gcd
    while x < 0:
        x += b_
        y -= a_
    while x > b_:
        x -= b_
        y += a_
    return (x, y)


# Xゲートの再実装
def my_x(self, qreg, cregs=None, cregs_value=None):
    if cregs is not None and cregs_value is not None:
        self.x(qreg).c_if(cregs, cregs_value)
    else:
        self.x(qreg)


# CXゲートの再実装
def my_cx(self, control, target, cregs=None, cregs_value=None):
    if cregs is not None and cregs_value is not None:
        self.cx(control, target).c_if(cregs, cregs_value)
    else:
        self.cx(control, target)


# CCXゲートの再実装
def my_ccx(self, control1, control2, target, cregs=None, cregs_value=None):
    if cregs is not None and cregs_value is not None:
        self.ccx(control1, control2, target).c_if(cregs, cregs_value)
    else:
        self.ccx(control1, control2, target)


# U1ゲートの再実装
def my_u1(self, angle, qreg, cregs=None, cregs_value=None):
    if cregs is not None and cregs_value is not None:
        self.u1(angle, qreg).c_if(cregs, cregs_value)
    else:
        self.u1(angle, qreg)


# CU1ゲートの再実装
def my_cu1(self, angle, control, qreg, cregs=None, cregs_value=None):
    if cregs is not None and cregs_value is not None:
        self.cu1(angle, control, qreg).c_if(cregs, cregs_value)
    else:
        self.cu1(angle, control, qreg)


# Hゲートの再実装
def my_h(self, qreg, cregs=None, cregs_value=None):
    if cregs is not None and cregs_value is not None:
        self.h(qreg).c_if(cregs, cregs_value)
    else:
        self.h(qreg)


# CSWAPゲートの再実装
def my_cswap(self, control, target1, target2, cregs=None, cregs_value=None):
    self.my_cx(target1, target2, cregs, cregs_value)
    self.my_ccx(control, target2, target1, cregs, cregs_value)
    self.my_cx(target1, target2, cregs, cregs_value)


# 与えられたすべてのビットを測定
def batch_measure(self, qregs, cregs):
    qregs, cregs = list(qregs), list(cregs)
    if(len(qregs) > len(cregs)):
        assert("Going to measure more quantum bits than classical bits")
    for qrj, crj in zip(qregs, cregs):
        self.measure(qrj, crj)


# 与えられたすべてのビットにCSWAPゲートを作用
def batch_cswap(self, control, qregs1, qregs2, cregs=None, cregs_value=None):
    qregs1, qregs2 = list(qregs1), list(qregs2)
    for qr1j, qr2j in zip(qregs1, qregs2):
        self.my_cswap(control, qr1j, qr2j, cregs, cregs_value)


# 制御制御位相回転ゲート
def ccu1(self, angle, control1, control2, target, cregs=None, cregs_value=None):
    self.my_cu1(angle / 2, control2, target, cregs, cregs_value)
    self.my_cx(control1, control2, cregs, cregs_value)
    self.my_cu1(-angle / 2, control2, target, cregs, cregs_value)
    self.my_cx(control1, control2, cregs, cregs_value)
    self.my_cu1(angle / 2, control1, target, cregs, cregs_value)


# 量子フーリエ変換ゲート
def qft(self, qregs, cregs=None, cregs_value=None):
    qregs = list(qregs)
    for j, qrj in reversed(list(enumerate(qregs))):
        self.my_h(qrj, cregs, cregs_value)
        for k, qrk in reversed(list(enumerate(qregs[:j]))):
            self.my_cu1(math.pi/float(2**(j - k)),
                        qrk, qrj, cregs, cregs_value)


# 逆量子フーリエ変換ゲート
def iqft(self, qregs, cregs=None, cregs_value=None):
    qregs = list(qregs)
    for j, qrj in enumerate(qregs):
        for k, qrk in enumerate(qregs[:j]):
            self.my_cu1(math.pi/float(-2**(j - k)),
                        qrk, qrj, cregs, cregs_value)
        self.my_h(qrj, cregs, cregs_value)


# ΦADD(a)ゲート qregsはフーリエ変換されたもの
def phiadd(self, a, qregs, cregs=None, cregs_value=None):
    qregs = list(qregs)
    bools = list(reversed([bool(i) for i in bin_digit(a)]))
    bools.extend([False] * (len(qregs) - len(bools)))
    for j, qrj in reversed(list(enumerate(qregs))):
        for k in reversed(range(j + 1)):
            if bools[k]:
                self.my_u1(math.pi/float(2**(j - k)), qrj, cregs, cregs_value)


# ΦADD(a)ゲートの逆演算 qregsはフーリエ変換されたもの
def inv_phiadd(self, a, qregs, cregs=None, cregs_value=None):
    qregs = list(qregs)
    bools = list(reversed([bool(i) for i in bin_digit(a)]))
    bools.extend([False] * (len(qregs) - len(bools)))
    for j, qrj in enumerate(qregs):
        for k in range(j + 1):
            if bools[k]:
                self.my_u1(-math.pi/float(2**(j - k)), qrj, cregs, cregs_value)


# CΦADD(a)ゲート
def cphiadd(self, a, control, qregs, cregs=None, cregs_value=None):
    qregs = list(qregs)
    bools = list(reversed([bool(i) for i in bin_digit(a)]))
    bools.extend([False] * (len(qregs) - len(bools)))
    for j, qrj in reversed(list(enumerate(qregs))):
        for k in reversed(range(j + 1)):
            if bools[k]:
                self.my_cu1(math.pi/float(2**(j - k)),
                            control, qrj, cregs, cregs_value)


# CΦADD(a)^-1ゲート
def inv_cphiadd(self, a, control, qregs, cregs=None, cregs_value=None):
    qregs = list(qregs)
    bools = list(reversed([bool(i) for i in bin_digit(a)]))
    bools.extend([False] * (len(qregs) - len(bools)))
    for j, qrj in enumerate(qregs):
        for k in range(j + 1):
            if bools[k]:
                self.my_cu1(-math.pi/float(2**(j - k)),
                            control, qrj, cregs, cregs_value)


# CCΦADD(a)ゲート
def ccphiadd(self, a, control1, control2, qregs, cregs=None, cregs_value=None):
    qregs = list(qregs)
    bools = list(reversed([bool(i) for i in bin_digit(a)]))
    bools.extend([False] * (len(qregs) - len(bools)))
    for j, qrj in reversed(list(enumerate(qregs))):
        for k in reversed(range(j + 1)):
            if bools[k]:
                self.ccu1(math.pi/float(2**(j - k)), control1,
                          control2, qrj, cregs, cregs_value)


# CCΦADD(a)^-1ゲート
def inv_ccphiadd(self, a, control1, control2, qregs, cregs=None, cregs_value=None):
    qregs = list(qregs)
    bools = list(reversed([bool(i) for i in bin_digit(a)]))
    bools.extend([False] * (len(qregs) - len(bools)))
    for j, qrj in enumerate(qregs):
        for k in range(j + 1):
            if bools[k]:
                self.ccu1(-math.pi/float(2**(j - k)), control1,
                          control2, qrj, cregs, cregs_value)


# CCΦADD(a)MOD(N)ゲート 補助ビットが必要
def ccphiaddmod(self, a, N, control1, control2, qregs, ancilla, cregs=None, cregs_value=None):
    qregs = list(qregs)
    bools = list(reversed([bool(i) for i in bin_digit(a)]))
    self.ccphiadd(a, control1, control2, qregs, cregs, cregs_value)
    self.inv_phiadd(N, qregs, cregs, cregs_value)
    self.iqft(qregs, cregs, cregs_value)
    self.my_cx(qregs[-1], ancilla, cregs, cregs_value)
    self.qft(qregs, cregs, cregs_value)
    self.cphiadd(N, ancilla, qregs, cregs, cregs_value)
    self.inv_ccphiadd(a, control1, control2, qregs, cregs, cregs_value)
    self.iqft(qregs, cregs, cregs_value)
    self.my_x(qregs[-1], cregs, cregs_value)
    self.my_cx(qregs[-1], ancilla, cregs, cregs_value)
    self.my_x(qregs[-1], cregs, cregs_value)
    self.qft(qregs, cregs, cregs_value)
    self.ccphiadd(a, control1, control2, qregs, cregs, cregs_value)


# CCΦADD(a)MOD(N)^-1ゲート 補助ビットが必要
def inv_ccphiaddmod(self, a, N, control1, control2, qregs, ancilla, cregs=None, cregs_value=None):
    qregs = list(qregs)
    bools = list(reversed([bool(i) for i in bin_digit(a)]))
    self.inv_ccphiadd(a, control1, control2, qregs, cregs, cregs_value)
    self.iqft(qregs, cregs, cregs_value)
    self.my_x(qregs[-1], cregs, cregs_value)
    self.my_cx(qregs[-1], ancilla, cregs, cregs_value)
    self.my_x(qregs[-1], cregs, cregs_value)
    self.qft(qregs, cregs, cregs_value)
    self.ccphiadd(a, control1, control2, qregs, cregs, cregs_value)
    self.inv_cphiadd(N, ancilla, qregs, cregs, cregs_value)
    self.iqft(qregs, cregs, cregs_value)
    self.my_cx(qregs[-1], ancilla, cregs, cregs_value)
    self.qft(qregs, cregs, cregs_value)
    self.phiadd(N, qregs, cregs, cregs_value)
    self.inv_ccphiadd(a, control1, control2, qregs, cregs, cregs_value)


# CMULT(a)MOD(N)ゲート qregs2の最上位は0でなければならない 補助ビットが必要
def cmultmod(self, a, N, control, qregs1, qregs2, ancilla, cregs=None, cregs_value=None):
    qregs1, qregs2 = list(qregs1), list(qregs2)
    self.qft(qregs2, cregs, cregs_value)
    for j, qr1j in enumerate(qregs1):
        self.ccphiaddmod((a * 2**j) % N, N, control, qr1j,
                         qregs2, ancilla, cregs, cregs_value)
    self.iqft(qregs2, cregs, cregs_value)


# CMULT(a)MOD(N)ゲート qregs2の最上位は0でなければならない 補助ビットが必要
def inv_cmultmod(self, a, N, control, qregs1, qregs2, ancilla, cregs=None, cregs_value=None):
    qregs1, qregs2 = list(qregs1), list(qregs2)
    self.qft(qregs2, cregs, cregs_value)
    for j, qr1j in reversed(list(enumerate(qregs1))):
        self.inv_ccphiaddmod((a * 2**j) % N, N, control,
                             qr1j, qregs2, ancilla, cregs, cregs_value)
    self.iqft(qregs2, cregs, cregs_value)


# C-Uaゲート aとNが互いに素な場合のみうまく動く qregsより2qubit多い補助ビット列が必要
def cua(self, a, N, control, qregs, ancillae, cregs=None, cregs_value=None):
    qregs, ancillae = list(qregs), list(ancillae)
    assert math.gcd(a, N) == 1, "a and N must be co-prime."
    assert len(ancillae) <= len(
        qregs) + 2, "quantum registers must be at least 2 qubit more than auxillliary q-bits"
    self.cmultmod(a, N, control, qregs,
                  ancillae[:-1], ancillae[-1], cregs, cregs_value)
    self.batch_cswap(control, qregs, ancillae, cregs, cregs_value)
    (a_inv, _) = solve_linear_indeterminate(a, N)
    self.inv_cmultmod(a_inv, N, control, qregs,
                      ancillae[:-1], ancillae[-1], cregs, cregs_value)


# 位相推定用のデータを得る
# Nは素因数分解の対象となる二つの奇素数の積 aは素因数分解に使用する数でa<NかつaとNは互いに素である必要がある
def qpe(self, a, N, control, qregs, ancillae, cregs):
    qregs, ancillae = list(qregs), list(ancillae)
    zero_and_one = [0, 1]
    self.my_x(qregs[0])
    for j in range(2 * len(qregs)):
        self.reset(control)
        self.my_h(control)
        self.cua(a**(2**(2 * len(qregs) - j - 1)) %
                 N, N, control, qregs, ancillae)
        for digit_list in list(product(zero_and_one, repeat=j)):
            cregs_value = inv_bin_digit(digit_list)
            for k, digit_k in enumerate(digit_list):
                if digit_k:
                    self.my_u1(math.pi / 2**(j - k),
                               control, cregs, cregs_value)
        self.my_h(control)
        self.measure(control, cregs[j])


# 作成した関数をメソッドとして追加
new_methods = [my_x, my_cx, my_ccx, my_u1, my_h, my_cu1, my_cswap,
               qft, iqft, batch_cswap, batch_measure, ccu1,
               phiadd, inv_phiadd, cphiadd, inv_cphiadd, ccphiadd, inv_ccphiadd, ccphiaddmod, inv_ccphiaddmod,
               cmultmod, inv_cmultmod, cua, qpe]

for method in new_methods:
    setattr(QuantumCircuit, method.__name__, method)

if __name__ == "__main__":
    # 素因数分解したい数Nに対してN<2^nであるとき素因数分解には2n+3qubit必要
    # 例えば15を素因数分解するなら15<2^4より11qubit必要
    size = 11
    qp = QuantumProgram()
    # 量子ビットの確保
    q = qp.create_quantum_register("q", size)
    # 出力のための古典ビットを確保 2n bit必要
    c = qp.create_classical_register("c", size-3)
    circ = qp.create_circuit("circ", [q], [c])
    qp.get_circuit("circ")
    q = list(q)
    ctrl = q[0]
    qregs = q[1:5]
    auxes = q[5:11]
    # 8を使って15を素因数分解
    # 結果から読み取れるピークの周期で2^(2n)で割った数をmとすると
    # gcd(a^(m/2)+1, N)とgcd(a^(m/2)-1, N)が素因数分解の解
    # 例えばこの場合ピークの周期が2^8/1000000(二進数)=4
    # gcd(8^(4/2)+1, 15) = 5, gcd(8^(4/2)-1, 15) = 3より15=5*3と素因数分解できる
    # aの取り方によってはうまくいかない
    circ.qpe(8, 15, ctrl, qregs, auxes, c)
    simulate = qp.execute(
        ["circ"], backend="local_qasm_simulator", shots=1024, timeout=10000)
    data = simulate.get_counts("circ")
    print(data)
