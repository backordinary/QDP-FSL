# https://github.com/hmwri/Processing/blob/341ad9eb842d8b1e43278a4dc308c76eb50ff5c7/ryoshi/data/ryoshiLibrary.py
import json
import matplotlib.pyplot as plt
import qiskit.result
from scipy.optimize import minimize, Bounds
from qiskit import QuantumCircuit, QuantumRegister, Aer, transpile, ClassicalRegister, IBMQ
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import C3XGate, C4XGate
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
import sys
import enum


class AuthorType(enum.Enum):
    User = "u_"
    Compiler = "c_"


class ryoshiRegister:
    targetCircuit: QuantumCircuit
    classicalRegister: ClassicalRegister

    def __init__(self, circuit: QuantumCircuit, register: QuantumRegister, author: AuthorType, makeClassic=True):
        self.nowValues = [0]
        self.register = register
        self.targetCircuit = circuit
        if makeClassic:
            self.classicalRegister = ClassicalRegister(register.size, register.name + "_c")
            self.targetCircuit.add_register(self.classicalRegister)
        self.name = register.name
        self.size = register.size
        self.author = author

    def IsAllBitSame(self, position):
        result = 0
        for v in self.nowValues:
            if v & 1 << position:
                result += 1
        if result == 0 or result == len(self.nowValues):
            print("true", position)
            return True
        return False

    def InvertBit(self, index, position):
        conditionBits = []
        notBits = []
        for i in range(0, self.register.size):
            if position != i and not self.IsAllBitSame(i):
                conditionBits.append(i)
                if not (self.nowValues[index] & 1 << i):
                    self.targetCircuit.x(self.register[i])
                    notBits.append(i)
        if len(conditionBits) == 1:
            self.targetCircuit.cx(self.register[conditionBits[0]], self.register[position])
        elif len(conditionBits) == 2:
            self.targetCircuit.ccx(self.register[conditionBits[0]], self.register[conditionBits[1]],
                                   self.register[position])
        elif len(conditionBits) == 3:
            registers = []
            for n in conditionBits:
                registers.append(self.register[n])
            registers.append(self.register[position])
            self.targetCircuit.append(C3XGate(), registers)
        elif len(conditionBits) == 4:
            registers = []
            for n in conditionBits:
                registers.append(self.register[n])
            registers.append(self.register[position])
            self.targetCircuit.append(C4XGate(), registers)
        else:
            registers = []
            for n in conditionBits:
                registers.append(self.register[n])
            self.targetCircuit.mcx(registers, self.register[position])
        for p in notBits:
            self.targetCircuit.x(self.register[p])
        print("invert:from:" + str(self.nowValues[index]) + "to;" + str(self.nowValues[index] ^ 1 << position))
        self.nowValues[index] = self.nowValues[index] ^ 1 << position

    def X(self, index):
        self.targetCircuit.x(self.register[index])
        for i in range(len(self.nowValues)):
            self.nowValues[i] = self.nowValues[i] ^ 2 ** index

    def H(self, index):
        self.targetCircuit.h(self.register[index])
        self.nowValues.append(2 ** index)
        self.hindex = index

    def AllH(self):
        self.targetCircuit.h(self.register)

    def Entangle2(self, num1, num2):
        if num1 == num2:
            print("num1 and num2 are same value")
            return
        minNum = min(num1, num2)
        maxNum = max(num1, num2)
        if list(bin(maxNum - minNum)).count('1') == 1 and list(bin(minNum ^ maxNum)).count('1') == 1:
            print("差が2のn乗")
            self.H(GetBitDigit(maxNum - minNum))
            if minNum == 0:
                print("差が2のn乗でかつ一方が0であったためHを一回適応し終了")
                return
            if self.nowValues[0] ^ minNum == self.nowValues[1] ^ maxNum:
                self.applyX(minNum)
                return
            self.applyCX(minNum, maxNum)
            return
        # どちらかが2のnじょうのときHゲートでよせる
        if list(bin(minNum)).count('1') == 1:
            self.H(GetBitDigit(minNum))
            self.applyCX(maxNum, minNum)
        elif list(bin(maxNum)).count('1') == 1:
            self.H(GetBitDigit(maxNum))
            self.applyCX(minNum, maxNum)
        else:
            self.H(0)
            self.applyCX(minNum, maxNum)

    def writeValue(self, num: int):
        formatString = '0' + str(self.size) + 'b'
        for i, b in enumerate(reversed(list(format(num, formatString)))):
            if b == '1':
                self.targetCircuit.x(self.register[i])

    def applyX(self, num1):
        print("移動する距離が同じ")
        for i in range(0, self.register.size):
            if (self.nowValues[0] ^ num1) & 1 << i:
                self.X(i)
        return

    def applyCX(self, num1, num2):
        self.InvertBits(0, num1)
        self.InvertBits(1, num2)

    def InvertBits(self, index, num):
        if self.nowValues[index] == num:
            print("same")
            return
        for i in range(self.register.size):
            print(i)
            if (self.nowValues[index] ^ num) & (1 << i):
                if not self.nowValues[index] ^ 1 << i in self.nowValues:
                    self.InvertBit(index, i)
                    self.InvertBits(index, num)
                    return

    def measure(self):
        self.targetCircuit.measure(self.register, self.classicalRegister)


class ryoshiCircuit:
    targetCircuit: QuantumCircuit
    registers: dict[ryoshiRegister]
    result: qiskit.result.Result
    measured: list[str]

    def __init__(self, name):

        print("start")
        self.name = name
        self.targetCircuit = QuantumCircuit(name=name)
        self.registers = {}
        self.measured = []
        self.result = None

    def appendRegister(self, key, register):
        self.registers[key] = register
        self.targetCircuit.add_register(register.register)

    def makeRegister(self, name, bit, author: AuthorType = AuthorType.User):
        qr = QuantumRegister(bit, name=author.value + name)
        register = ryoshiRegister(self.targetCircuit, qr, author, author == AuthorType.User)
        self.appendRegister(name, register)

    def makeCircuit(self):
        qrs = []
        for rr in self.registers.values():
            qrs.append(rr.register)
            qrs.append(rr.classicalRegister)
        self.targetCircuit = QuantumCircuit(*qrs, name=self.name)
        for rr in self.registers.values():
            rr.targetCircuit = self.targetCircuit

    def AllH(self, name):
        self.registers[name].AllH()

    def equal(self, register1: str, register2: str, help: str, result: str):
        self.targetCircuit.barrier()
        rr1 = self.registers[register1]
        rr2 = self.registers[register2]
        rrr = self.registers[result]
        if rr1.size > rr2.size :
            tmp = rr2
            rr2 = rr1
            rr1 = tmp

        rrc = self.registers[help]
        for i in range(rr2.size):
            if i < rr1.size:
                self.targetCircuit.cx(rr1.register[i], rrc.register[i])
            self.targetCircuit.cx(rr2.register[i], rrc.register[i])
        for i in range(rrc.size):
            self.targetCircuit.x(rrc.register[i])

        controlls = [c for c in rrc.register]
        self.targetCircuit.mcx(controlls, rrr.register[0])

    def inv_equal(self, register1: str, register2: str, help: str, result: str):
        self.targetCircuit.barrier()
        rr1 = self.registers[register1]
        rr2 = self.registers[register2]
        rrr = self.registers[result]
        rrc = self.registers[help]
        if rr1.size > rr2.size :
            tmp = rr2
            rr2 = rr1
            rr1 = tmp
        controlls = [c for c in rrc.register]
        self.targetCircuit.mcx(controlls, rrr.register[0])

        for i in range(rrc.size):
            self.targetCircuit.x(rrc.register[i])

        for i in range(rr2.size):
            self.targetCircuit.cx(rr2.register[i], rrc.register[i])
            if i < rr1.size:
                self.targetCircuit.cx(rr1.register[i], rrc.register[i])

    def inv_notequal(self, register1: str, register2: str, help: str, result: str):
        rrr = self.registers[result]
        self.targetCircuit.x(rrr.register[0])
        self.inv_equal(register1, register2, help, result)

    def notequal(self, register1: str, register2: str, help: str, result: str):
        rrr = self.registers[result]
        self.equal(register1, register2, help, result)
        self.targetCircuit.x(rrr.register[0])

    def plusEqual(self,register1:str,register2:str):
        self.targetCircuit.barrier()
        rr1 = self.registers[register1]
        rr2 = self.registers[register2]
        for i in range(rr2.size) :
            target = rr2.register[i]
            for j in reversed(range(i,rr1.size)):
                controlls = []
                for k in range(i,j):
                    controlls.append(rr1.register[k])
                controlls.append(target)
                self.targetCircuit.mcx(controlls,rr1.register[j])

    def minusEqual(self,register1:str,register2:str):
        self.targetCircuit.barrier()
        rr1 = self.registers[register1]
        rr2 = self.registers[register2]
        for i in reversed(range(rr2.size)) :
            target = rr2.register[i]
            for j in range(i,rr1.size):
                controlls = []
                for k in range(i,j):
                    controlls.append(rr1.register[k])
                controlls.append(target)
                self.targetCircuit.mcx(controlls,rr1.register[j])

    def And(self, register1: str, register2: str, result: str):
        rr1 = self.registers[register1]
        rr2 = self.registers[register2]
        rrr = self.registers[result]
        if not rr1.size == rr2.size == rrr.size == 1:
            print("ERROR rr1 and rr2 are not bool")
            return
        self.targetCircuit.ccx(rr1.register, rr2.register, rrr.register)

    def inv_And(self, register1: str, register2: str, result: str):
        rr1 = self.registers[register1]
        rr2 = self.registers[register2]
        rrr = self.registers[result]
        if not rr1.size == rr2.size == rrr.size == 1:
            print("ERROR rr1 and rr2 are not bool")
            return
        self.targetCircuit.ccx(rr1.register, rr2.register, rrr.register)

    def Or(self, register1: str, register2: str, result: str):
        rr1 = self.registers[register1]
        rr2 = self.registers[register2]
        rrr = self.registers[result]
        if not rr1.size == rr2.size == rrr.size == 1:
            print("ERROR rr1 and rr2 are not bool")
            return
        self.targetCircuit.x(rr1.register)
        self.targetCircuit.x(rr2.register)
        self.targetCircuit.ccx(rr1.register, rr2.register, rrr.register)
        self.targetCircuit.x(rr1.register)
        self.targetCircuit.x(rr2.register)
        self.targetCircuit.x(rrr.register)

    def inv_Or(self, register1: str, register2: str, result: str):
        rr1 = self.registers[register1]
        rr2 = self.registers[register2]
        rrr = self.registers[result]
        if not rr1.size == rr2.size == rrr.size == 1:
            print("ERROR rr1 and rr2 are not bool")
            return
        self.targetCircuit.x(rrr.register)
        self.targetCircuit.x(rr1.register)
        self.targetCircuit.x(rr2.register)
        self.targetCircuit.ccx(rr1.register, rr2.register, rrr.register)
        self.targetCircuit.x(rr1.register)
        self.targetCircuit.x(rr2.register)

    def Not(self, target: str):
        rr1 = self.registers[target]
        self.targetCircuit.x(rr1.register)

    def write(self, name, value):
        self.registers[name].writeValue(value)

    def diffuser(self, *registers):
        self.targetCircuit.barrier()
        qRs = [self.registers[r].register for r in registers]
        qRbits = []
        for r in qRs:
            self.targetCircuit.h(r)
            self.targetCircuit.x(r)
            for b in r:
                qRbits.append(b)
        self.targetCircuit.h(qRbits[0])
        self.targetCircuit.mct(qRbits[1:], qRbits[0])
        self.targetCircuit.h(qRbits[0])
        for r in qRs:
            self.targetCircuit.x(r)
            self.targetCircuit.h(r)

    def mark(self, *registers):
        self.targetCircuit.barrier()
        qRs = [self.registers[r].register for r in registers]
        qRbits = []
        for r in qRs:
            for b in r:
                qRbits.append(b)
        if len(qRbits) == 1:
            self.targetCircuit.z(qRbits[0])
        else:
            self.targetCircuit.h(qRbits[0])
            self.targetCircuit.mct(qRbits[1:], qRbits[0])
            self.targetCircuit.h(qRbits[0])

    def Entangle2(self, name: str, num1, num2):
        self.registers[name].Entangle2(num1, num2)

    def measure(self, *registers):
        for r in registers:
            self.measured.append(r);
            self.registers[r].measure()

    def exe_sim(self, shots):
        print("シミュレータで実行します")
        simulator = Aer.get_backend('qasm_simulator')
        circuit = transpile(self.targetCircuit, backend=simulator)
        print("トランスパイル完了")
        sim_job = simulator.run(circuit, shots=shots)
        print("実行完了")
        sim_result = sim_job.result()
        self.result = sim_result

    def exe_actual(self, shots):
        print("実機で実行します")
        IBMQ.load_account()
        print("トークン認証完了しました")
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend_list = provider.backends()
        backend: qiskit.providers.Backend = least_busy(backend_list)
        print("バックエンド準備完了 backend" + str(backend.version))
        circuit = transpile(self.targetCircuit, backend=backend)
        print("トランスパイル完了")
        print("実行開始")
        job = backend.run(circuit, shots=shots)
        job_monitor(job, interval=2)
        print("実行完了")
        result = job.result()
        self.result = result

    def get_result(self, *names, debug=False):
        shots = self.result.to_dict()['results'][0]['shots']
        cregs = self.result.to_dict()['results'][0]['header']['creg_sizes']
        if len(names) == 0:
            names = self.measured
        cregnames = [c[0] for c in cregs]
        counts = self.result.get_counts()
        results = {}
        results2 = {}

        for key, c in counts.items():
            bits = key.split()
            bits.reverse()
            keyname = ""
            for i, bit in enumerate(bits):
                bit = str(int(bit, 2))
                cregname = cregnames[i][2:len(cregnames[i]) - 2]
                if cregname in names:
                    keyname += cregname + "=" + bit + ","
            keyname = keyname[:len(keyname) - 1]
            if keyname in results2:
                results2[keyname] += c
            else:
                results2[keyname] = c

        # for key, c in counts.items():
        #     bits = key.split()
        #     bits.reverse()
        #     for i, bit in enumerate(bits):
        #         bit = str(int(bit, 2))
        #         cregname = cregnames[i][2:len(cregnames[i]) - 2]
        #         if cregname not in results:
        #             results[cregname] = {}
        #         if bit not in results[cregname]:
        #             results[cregname][bit] = 0
        #         results[cregname][bit] += c
        #
        # for key1, r in results.items():
        #     for key2, n in r.items():
        #         results[key1][key2] = n / shots
        # print(results)

        sortedResult = sorted(results2.items(), key=lambda x: x[1], reverse=True)
        result = {'shots': shots, 'result': sortedResult}
        string = json.dumps(result)
        f = open('result.json', 'w')
        f.write(string)
        f.close()
        if debug:
            plot_histogram(counts)
            plt.tight_layout()
            plt.show()
        print("回路図生成中")
        self.targetCircuit.draw('mpl', filename="circuit.png",scale=0.4)
        print("完了")

    # positon番目のビットがすべて同じか


def GetBitDigit(number):
    return len(bin(number)) - 3


def test(registerSize):
    h = int(2 ** registerSize / 2) - 1
    w = 2 ** registerSize
    image = False

    errorString = ""
    counts_list = []
    circuits = []
    for i in range(2 ** registerSize):
        for j in range(2 ** registerSize):
            if (i > j):
                continue
            if (i == j):
                continue
            print("j:", j, "i:", i)
            rc = ryoshiCircuit(name='circuit_' + str(i) + '|' + str(j))
            rc.makeRegister(bit=registerSize, name="aiueo")
            rc.makeCircuit()
            rc.registers[0].Entangle2(i, j)
            rc.targetCircuit.measure_all()
            # シミュレータにはショット数の制限がないので、時間の許す限りいくらでも大きい値を使っていい
            shots = 10
            simulator = Aer.get_backend('qasm_simulator')
            # 実習と同じく transpile() - 今は「おまじない」と思ってよい
            circuit = transpile(rc.targetCircuit, backend=simulator)
            # シミュレータもバックエンドと同じように振る舞うので、runメソッドで回路とショット数を受け取り、ジョブオブジェクトを返す
            sim_job = simulator.run(circuit, shots=shots)
            # シミュレータから渡されたジョブオブジェクトは実機のジョブと全く同じように扱える
            sim_result = sim_job.result()
            counts = sim_result.get_counts()
            counts_list.append(counts)
            circuits.append(circuit)
            formatString = '0' + str(registerSize) + 'b'
            if format(i, formatString) in counts and format(j, formatString) in counts:
                print("OK")
            else:
                print("ERROR!!:", "j:", j, "i:", i, ":", counts)
                errorString += "ERROR!!:" + "j:" + str(j) + "i:" + str(i) + ":" + str(counts) + "\n"
    if (len(errorString) == 0):
        print("SUCCESS!")
    print(errorString)
    if not image:
        return
    fig, axs = plt.subplots(h, w, sharey=True, figsize=(w * 4, h * 2))
    for counts, circuit, ax in zip(counts_list, circuits, axs.reshape(-1)):
        ax.set_title(circuit.name)
        circuit.draw('mpl', ax=ax)
        ax.yaxis.grid(True)
    plt.tight_layout()
    fig2, axs2 = plt.subplots(h, w, sharey=True, figsize=(w * 2.5, h * 2.7))
    for counts, circuit, ax in zip(counts_list, circuits, axs2.reshape(-1)):
        ax.set_title(circuit.name)
        plot_histogram(counts, ax=ax)
        ax.yaxis.grid(True)
    plt.tight_layout()
    fig.savefig("circuit.png")
    fig2.savefig("result.png")
