# https://github.com/otbrown/qiskit-circuits/blob/846ad35c7f1ddc836c2d0948087f0752afddb907/qft.py
import argparse as ap
import timeit as ti
import qiskit as qk
import math as m
from qiskit import circuit
from qiskit.providers import aer
from mpi4py import MPI


def parse_args():
  parser = ap.ArgumentParser(description='Read in the number of qubits.')
  parser.add_argument('N_QUBITS',
    type=int,
    help='The number of qubits to be simulated. Must be in range 0 < N_QUBITS < 45.',
    choices=range(1,45)
  )

  return parser.parse_args()

def main():
  run_start = ti.default_timer()
  
  args = parse_args()
  
  qft = qk.circuit.library.QFT(
    num_qubits=args.N_QUBITS,
    approximation_degree=0,
    do_swaps=False,
    inverse=False,
    insert_barriers=False,
  )
  qft.save_state(label='statevector')
  
  sim = aer.AerSimulator(method='statevector')

  transp_qft = qk.transpile(qft, sim)
  N_GATES = transp_qft.size()

  COMM = MPI.COMM_WORLD
  RANK = COMM.Get_rank()

  if (RANK == 0):
    print("Running QFT circuit")
    print("  Number of qubits: ", args.N_QUBITS)
    print("  Number of gates: ", N_GATES)
    print()
  
  qft_start = ti.default_timer()

  result = sim.run(transp_qft, method='statevector', blocking_enable=True, blocking_qubits=28).result()

  COMM.Barrier()
  qft_stop = ti.default_timer()
  run_stop = ti.default_timer()

  if RANK == 0:
    print("Results:")
    print('  Run time = {:g} s'.format(run_stop - run_start))
    print('  QFT time = {:g} s'.format(qft_stop - qft_start))
    print('  Measured probability of |0..0> state = {:g}'.format(abs(result.get_statevector()[0])**2))
    print('  Calculated probability of |0..0> state = {:g}'.format(2**-args.N_QUBITS))

  MPI.Finalize()

if __name__ == "__main__":
  main()

