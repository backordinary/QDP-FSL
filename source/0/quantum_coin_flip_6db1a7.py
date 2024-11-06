# https://github.com/TamatiB/coding_academy/blob/1c8c86a767b78d2ef45de0819ee702298c405590/lesson_6/quantum_coin_flip.py
import random
import qiskit as qk
from qiskit import execute
from qiskit import Aer

def quantum_coin_flip():
  backend = Aer.get_backend("qasm_simulator")

  #create Quantum Register with 1 qubit
  qr = qk.QuantumRegister(1)
  #initialize a calssical register with the single bit
  cr = qk.ClassicalRegister(1)
  # create a  quantum circuit acting on the q register
  qc = qk.QuantumCircuit(qr,cr)


  print("Computer plays..")
  # Apply hadamard gate, which randomises the state
  qc.h(qr[0])

  humen = input("Would you like to flip? Yes(y) or No(n): ")
  if humen == "y":
    # X is a gate which is a qubit flip operator
    qc.x(qr[0])

  print("Computer plays again...")
  qc.h(qr[0])
  
###############################################
# You can use a real quantum computer if you have an load_account
# https://quantum-computing.ibm.com/signup
###############################################

  #qk.IBMQ.load_accounts()
  #backend = qk.providers.ibmq.least_busy(qk.IBMQ.backends(simulator=False))
  #print("We'll use the least busy device:",backend.name())
  qc.measure(qr,cr)
  job = execute(qc, backend = backend, shots = 1)

  results = job.result()
  counts = results.get_counts()

  if counts["0"] == 1:
    print("You lose. Soz")
  if counts["0"] == 0:
    print("You win, but it probably a mistake in the computer")


def flip_coin():
  return(random.randint(0,1))

def computer_play(coin_var):
  #chooses to flip or not flip, it it choses to flip(1) it flips
  flip_or_not = random.randint(0,1)
  if flip_or_not == 1:
    return(flip_coin()) #it calls the flip function to flip the coin
  else:
    return coin_var


def classical_coin_flip():
  coin = 1
  coin = computer_play(coin)
  humen = input("Would you like to flip? Yes(y) or No(n): ")
  if humen == "y":
    coin = flip_coin()
  coin = computer_play(coin)
  print(coin)
  if coin == 0:
    print("You lose. Soz")
  if coin == 1:
    print("You win! Well done")

game = input("Classical game (c) or Quantum game (q)?: ")
if game == "c":
  classical_coin_flip()
if game == "q":
  quantum_coin_flip()
