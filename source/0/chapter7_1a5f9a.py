# https://github.com/snuffkin/books/blob/6461ab69df1c0f06e0b9a26b47e52e93cc57212c/20190414_quantum_computer/python/chapter7.py
# -*- coding: utf-8 -*-
"""chapter7.ipynb

# 第7章のソースコード

## リスト7.1: API Token を設定

'MY_API_TOKEN'はご自身のAPI Tokenに書き換えて実行してください。
"""

from qiskit import IBMQ
IBMQ.enable_account('MY_API_TOKEN')

"""## リスト7.2: 量子コンピュータを実行するプログラム(実行する毎に結果は変化します)

'ibmqx4'がメンテナンス中で利用できなかったため、backendは'ibmq_16_melbourne'で実行しています。
"""

from qiskit import QuantumCircuit
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import execute
import time
from qiskit.providers.jobstatus import JOB_FINAL_STATES

# 量子回路の初期化
qr = QuantumRegister(2, 'q') # 量子レジスタを作成
cr = ClassicalRegister(2, 'c') # 古典レジスタを作成
circuit = QuantumCircuit(qr, cr) # レジスタを使い量子回路を初期化

# 量子回路の組み立て
circuit.h(qr[0]) # アダマール行列を適用
circuit.cx(qr[0], qr[1]) # CNOTを適用

# 測定
circuit.measure(qr, cr)

# 実行と結果取得
backend = IBMQ.get_backend('ibmq_16_melbourne') # デバイス指定 
job = execute(circuit, backend) # 量子プログラムを実行

# 実行が終了するまで待機
start_time = time.time()
job_status = job.status()
while job_status not in JOB_FINAL_STATES:
    time.sleep(10)
    job_status = job.status()
    print('%0.0f s' % float(time.time()-start_time))

result = job.result() # 結果を取得
print(result.get_counts(circuit)) # 結果をテキスト表示

"""## リスト7.4: ヒストグラム表示"""

from qiskit.tools.visualization import plot_histogram

# ヒストグラム表示
plot_histogram(job.result().get_counts(circuit))