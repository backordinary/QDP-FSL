# https://github.com/keigo1216/QC4U/blob/b4fc04b24f96935a6dbd77cd65fbd6f1b59af391/src/server/util.py
from qiskit import QuantumCircuit, QuantumRegister, Aer, ClassicalRegister, transpile 
import numpy as np
import matplotlib.pyplot as plt

#qcの状態ベクトルの表示
def sim_state(qc,disp=True):
  #Aerはシミュレータに対するプログラムセットで、get_backendでstatevector_simulatorというシミュレータを取ってくるということ
  sim = Aer.get_backend('statevector_simulator')
  qc = transpile(qc, backend=sim)
  #qcに対してシミュレータを走らせて結果を出し、resで受け取る
  res = sim.run(qc).result()
  #resultとして出て来る様々な結果からstatevectorのみを取り出して、stateに格納する
  state = res.data()["statevector"]
  #各状態の確率を抽出
  p_00 = np.round(abs(state[0])**2*100, 2).real
  p_01 = np.round(abs(state[1])**2*100, 2).real
  p_10 = np.round(abs(state[2])**2*100, 2).real
  p_11 = np.round(abs(state[3])**2*100, 2).real
  if disp == True:
    #qcの状態ベクトルをケットベクトル表示に変換する
    #ket = matrix_to_qubit(np.array(state)[:, np.newaxis])
    #print(ket)
    print(np.round(state[0], 3), '|00> +', np.round(state[1], 3), '|01> +', np.round(state[2], 3), '|10> +', np.round(state[3], 3), '|11>')
    #print(np.round(state[0]**2*100, 2).real, np.round(state[1]**2*100, 2).real, np.round(state[2]**2*100, 2), np.round(state[3]**2*100, 2))
    print('|00> :', p_00, '% \t', end="")
    show_graph(p_00)
    print('|01> :', p_01, '% \t', end="")
    show_graph(p_01)
    print('|10> :', p_10, '% \t', end="")
    show_graph(p_10)
    print('|11> :', p_11, '% \t', end="")
    show_graph(p_11)
    print('--------------------------------------------------------------------------------')
  return state


#測定（量子ビットの頻度と履歴を出力）
def sim_state_experiment(qc, shots):
  sim = Aer.get_backend('qasm_simulator')
  qc = transpile(qc, backend=sim)
  #memory=Trueに設定する
  res = sim.run(qc, shots=shots, memory=True).result()
  ans = res.get_counts()
  mem = res.get_memory()
  return ans, mem

#量子ビットの履歴(2進数)をスロット番号（自然数）の履歴に変換
def slot_memory(mem, shots):
  slot = np.zeros(shots, dtype=int)
  for j in range(shots):
    slot[j] = int(mem[j], base=2) + 1
  return slot

#スロット番号からチップ数の増加を設定
def chip_increment(slot1, slot2, slot3):
  if   slot1==1 and slot2==1 and slot3==1:inc= 4
  elif slot1==2 and slot2==2 and slot3==2:inc= 5
  elif slot1==3 and slot2==3 and slot3==3:inc= 5
  elif slot1==4 and slot2==4 and slot3==4:inc= 4
  elif slot1==1 and slot2==2 and slot3==3:inc= 6
  elif slot1==2 and slot2==3 and slot3==4:inc= 6
  elif slot1==4 and slot2==3 and slot3==2:inc= 6
  elif slot1==3 and slot2==2 and slot3==1:inc= 6
  else :
    inc=-1 
  return inc

def show_graph(number):
  for i in range(int(number)):
    print('*', end="")
  print()