# https://github.com/Digas-2/Dissertation/blob/bbfdfecde77374def384e3bb8ac79a520dc28e49/Codigo_Dissertacao/qgan_mimic.py
import numpy as np
import pandas as pd


from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import TwoLocal, UniformDistribution

from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QGAN
from qiskit.aqua.components.neural_networks import NumPyDiscriminator

def main():
    df = pd.read_csv('../Datasets/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv')
    df = df.head(999)
    seed = 71
    N = 1000
    bounds = np.array([0., 3.])
    num_qubits = [2]
    k = len(num_qubits)
    num_epochs = 10
    batch_size = 100

    #real_data = df.to_numpy()
    real_data = np.random.lognormal(mean=1, sigma=1, size=N)
    print(real_data)
    #print(real_data.ndim())
    qgan = QGAN(real_data, bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None)
    qgan.seed = 1
    quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                   seed_transpiler=seed, seed_simulator=seed)

    entangler_map = [[0, 1]]
    init_dist = UniformDistribution(sum(num_qubits))
    var_form = TwoLocal(int(np.sum(num_qubits)), 'ry', 'cz', entanglement=entangler_map, reps=1)
    init_params = [3., 1., 0.6, 1.6]

    g_circuit = var_form.compose(init_dist, front=True)
    qgan.set_generator(generator_circuit=g_circuit, generator_init_params=init_params)
    # The parameters have an order issue that following is a temp. workaround
    qgan._generator._free_parameters = sorted(g_circuit.parameters, key=lambda p: p.name)
    # Set classical discriminator neural network
    discriminator = NumPyDiscriminator(len(num_qubits))
    qgan.set_discriminator(discriminator)

    result = qgan.run(quantum_instance)

    print('Training results:')
    for key, value in result.items():
        print(f'  {key} : {value}')

    #new_data.to_csv('../Output/mimic_output_diagnoses_icd_ctgan.csv',index=False)

if __name__ == "__main__":
    main()