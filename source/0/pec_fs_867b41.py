# https://github.com/QbitsCode/Variational-Quantum-Eigensolver/blob/3daea84dd5f4d1cc9e888f1df1bcb589593e955a/pec_fs.py
from time import time
from qiskit import Aer
from numpy import arange

from VQE_FS import VQE_fs, VQE_fs_test
from VQE_utils import build_molecule

He = build_molecule([('He', [0, 0, 0])], '6-31g', 0, 1, 'He')
H2O = build_molecule([('H', [-2.59626, 1.77639, 0.00000]), ('H', [-3.88959, 1.36040, -0.81444]), ('O', [-3.56626, 1.77639, 0])], 'sto3g', 0, 1, 'H2O')

lengths = []
energies = []

lb = 0.9
ub = 1.2
step = 0.1

Gstart = time()

for length in arange(lb, ub, step):
    H2 = build_molecule([('H', [0, 0, 0]), ('H', [0, 0, length])], 'sto-3g', 0, 1, 'H2')

    if length == lb:
        ω = -0.9
    else:
        init_angles = last_angles
        ω = last_energy

    myvqe = VQE_fs(
        H2,
        nlayers=1,
        wordiness=0,
        ω=ω,                                                                    #omega is the target zone of total energy, in Ha
        refstate='HF',
    )

    if length == lb:
        init_angles = [0.0 for _ in range(myvqe.nexc * myvqe.nlayers)]

    start = time()
    vqe_energy = myvqe.minimize_expval(init_angles, maxiter=1000)
    end = time()
    t = end - start

    print('Bond Length : ' + str(length) + '\n')
    print('Omega (Total Energy Shift) = ' + str(myvqe.omegatot) + ' Ha \n')
    print('refstate = ' + myvqe.refstate + '\n')
    print('final state = ' + myvqe.final_state + '\n')
    print('init', init_angles, '; ', myvqe.nlayers, 'layer(s)', '; ', myvqe.shots, 'shot(s)', ';', myvqe.backend, '; algo', myvqe.optimizer)
    print('optimization success ' + str(myvqe.success) + ' after', myvqe.niter, ' iterations ; final angles', myvqe.opt_angles)
    print('electronic energy ', vqe_energy, 'Ha')
    print('total energy : ', vqe_energy + myvqe.molecule.nuclear_energy, 'Ha')
    print('runtime : ' + str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str((t % 3600) % 60) + ' sec' + '\n')

    lengths.append(length)
    energies.append(vqe_energy + myvqe.molecule.nuclear_energy)
    print(lengths)
    print(energies)
    print('------------')
    last_angles = myvqe.opt_angles
    last_energy = vqe_energy + myvqe.molecule.nuclear_energy

Gend = time()

T = Gend - Gstart
print('Total Runtime : ' + str(T // 3600) + ' h ' + str((T % 3600) // 60) + ' min ' + str((T % 3600) % 60) + ' sec' + '\n')
