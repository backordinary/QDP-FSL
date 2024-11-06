# https://github.com/KazuyaManabe/Qiskit/blob/319e4583c915bd976fa11ce0ef52d7c71e98626b/QISKit_Text.py
#%%
from qiskit import *
from qiskit.tools.visualization import *

# %%26page
q = QuantumRegister(1) #1つの量子レジスタqの生成
c = ClassicalRegister(1) #1つの古典的レジスタcの生成
qc = QuantumCircuit(q,c) #量子回路qcの生成
qc.x(q[0])#量子ビットq[0]のビット反転演算,29page
qc.z(q[0])#量子ビットq[0]のビット反転演算,33page
qc.measure(q,c)

r = execute(qc, Aer.get_backend('qasm_simulator'), shots=100).result()
#量子回路を実行し、結果rに代入する
print("r: {}".format(r.get_counts()))
#量子回路名cnの量子プログラムの実行結果rからカウント結果を取得し表示する
rc = r.get_counts()
print("rc: {}".format(rc))

#%%量子回路名cnの量子プログラム実行によるヒストグラムデータ結果を表示する
plot_histogram(rc)
# %% 量子回路
circuit_drawer(qc)
