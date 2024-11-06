# https://github.com/awlaplace/Quantum/blob/359b943d749e695a577fd86f4a778918c907e6e4/Simon/Simon.py
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, execute


def quantum_simon_oracle(b: str, circuit: list) -> list:
    """
    nビット列 x に対して， |x>|0> を 入力として |x> |x \oplus b> を出力する関数
    ただし， x \oplus b で x と b の排他的論理和を表す
    |x>|0> → |x>|x> と，初めに１つめのレジスタの内容を2つめにコピーする
    次に， 後半 n ビットと b との排他的論理和で与える
    """
    n = len(b)

    # |x>|0> → |x>|x>
    for index in range(n):
        circuit.cx(index, index+ n)

    # |x>|x> → |x>|x \oplus b>
    for index in range(n):
        if b[n - 1 - index] == '1':
            target_index = index
            break
    for index in range(n):
        if b[index] == '1':
            circuit.cx(target_index, 2 * n - 1 - index)
    
    return circuit


def bdotz(b, z):
    accum = 0
    for i in range(len(b)):
        accum += int(b[i]) * int(z[i])

    return (accum % 2)


def main():
    b = "11000"
    n = len(b)
    simon_circuit = QuantumCircuit(n*2, n)

    # オラクルに入力する前にアダマールゲートを適用する
    simon_circuit.h(range(n))    
    
    # 可視性の向上のため、境界を挿入する
    simon_circuit.barrier()

    simon_circuit = quantum_simon_oracle(b, simon_circuit)

    # 可視性の向上のため、境界を挿入する
    simon_circuit.barrier()

    # 入力レジスターにアダマールゲートを適用する
    simon_circuit.h(range(n))

    # 量子ビットを測定する
    simon_circuit.measure(range(n), range(n))

    # ローカルシミュレーターを利用する
    backend = BasicAer.get_backend('qasm_simulator')
    shots = 1024
    results = execute(simon_circuit, backend=backend, shots=shots).result()
    counts = results.get_counts()

    for z in counts:
        print( '{}.{} = {} (mod 2)'.format(b, z, bdotz(b,z)) )


if __name__ == "__main__":
    main()
