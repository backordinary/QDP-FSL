# https://github.com/cct-blog/cct-seiya-kumada/blob/ccb160a8aabd0095e678f431763a2e7eaead1899/quantum_phase_estimation/main.py
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import quantum_phase_estimation as QPE
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import qiskit

# 第1レジスタのビット数
N_ENCODE = 5

if __name__ == "__main__":

    # 回路を作る。
    qc, theta = QPE.make_circuit(N_ENCODE)

    # θ=2*pi*φを与える（正解値）
    phase = np.random.rand()

    # 回路にパラメータを設定する。
    qc_parametrized = qc.bind_parameters({theta: phase})

    # シミュレータを選択する。
    backend = qiskit.Aer.get_backend('qasm_simulator')

    # 実行回数
    shots = 1024

    # 計算する。
    results = qiskit.execute(qc_parametrized, backend=backend, shots=shots).result()

    # 結果をとりだす。
    answer = results.get_counts()

    # 測定された位相を算出する。
    values = list(results.get_counts().values())
    keys = list(results.get_counts().keys())
    idx = np.argmax(list(results.get_counts().values()))
    ans = int(keys[idx], 2)
    print('Estimated state: {}'.format(ans))
    phase_estimated = ans / (2 ** N_ENCODE)

    # 正しい位相の値
    true_phase = phase / (2 * np.pi)

    print('True phase: {:.4f}'.format(true_phase))
    print('Estimated phase: {:.4f}'.format(phase_estimated))
    print('Diff: {:.4f}'.format(np.abs(true_phase - phase_estimated)))

    # ヒストグラムを描画して保存する。
    plot_histogram(answer, figsize=(20, 7))
    plt.savefig("./histogram.jpg")
