# https://github.com/royess/test-stab-entanglement/blob/b09d6a978e685002e461ea540a18302ee66a02d3/get_data.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import StabilizerState, Statevector, entropy, partial_trace, random_clifford
from qiskit import Aer
import numpy as np
import random
import h5py


def get_stab_list(stab, jlformat=True):
    stab_list = stab._data.stabilizer.to_labels()
    if jlformat:
        return '\n'.join([s.replace('I', '_') for s in stab_list])
    else:
        return '\n'.join(stab_list)


def sample_once(n_qubits):
    qc = random_clifford(n_qubits).to_circuit()
    stab_list = get_stab_list(StabilizerState(qc))
    backend = Aer.get_backend('statevector_simulator')
    job = backend.run(qc)
    result = job.result()
    outputstate = result.get_statevector(qc)
    leftend, rightend = sorted([random.randint(0,n_qubits-1), random.randint(0,n_qubits-1)])
    entanglement = entropy(partial_trace(Statevector(outputstate), range(leftend, rightend+1))) # note: cautions about order of qubits
    return stab_list, leftend, rightend, entanglement

def main():
    test_sizes = [3, 4, 6, 8]
    num_repeat = 200
    path = 'data'

    with h5py.File(f'{path}/stab_ent.h5', 'w') as f:
        for size in test_sizes:
            g = f.create_group(f"size{size}")
            dset_stab_list = g.create_dataset("stab_list", (num_repeat,), dtype=h5py.special_dtype(vlen=str))
            dset_leftend = g.create_dataset("leftend", (num_repeat,), dtype=int)
            dset_rightend = g.create_dataset("rightend", (num_repeat,), dtype=int)
            dset_entanglement = g.create_dataset("entanglement", (num_repeat,), dtype=float)
            for i_repeat in range(num_repeat):
                stab_list, leftend, rightend, entanglement = sample_once(size)
                dset_stab_list[i_repeat] = stab_list
                dset_leftend[i_repeat] = leftend
                dset_rightend[i_repeat] = rightend
                dset_entanglement[i_repeat] = entanglement



if __name__=='__main__':
    main()