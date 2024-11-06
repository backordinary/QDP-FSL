# https://github.com/marcobn/Quantum/blob/0b12d7ce39d589e0810c18f7b0eb183120c6ce38/BandTheory/redoband.py
""" This script finds the highest quality VQD run for each kpoint from
        all optimization runs thus far attempted,
        reruns expectations on a different quantum computer,
        then refines results with QPE.
    This file is meant to execute on a cloud device, so we need QPE to work
        after transpilation. There's a chance we'll need to do it ourselves...
"""
# IMPORTS
import os

import numpy as np
np.set_printoptions(3, suppress=True)
import qiskit

import qctb
import tightbinding
import tools


#                                SELECT SYSTEM
################################################################################
redofile = "json/ibmq_athens.bestruns.json"
datafile = "data/ibmq_athens.recap_QPU.data"
savefile = "json/ibmq_athens.recap_QPU.json"

QC = qctb.ibmq.CAL("ibmq_athens", qvm=False)



#                      RE-MEASURE EACH ENERGY WITH QC
################################################################################
VQD = qctb.vqd.load(redofile)
rVQD = qctb.vqd.VQD(VQD.H, QC, VQD.OPT, nopt=VQD.nopt)

for point in VQD.points:
    kpt = np.array(point['ka'])
    template = VQD.H.template(kpt, VQD.H.M-1)
    energyop = qctb.operator.ReciprocalOperator(VQD.H, kpt, VQD.ansatz)

    print ("Evaluating k =", kpt)

    for level in point['levels']:
        l = level['l']
        x = level['x']

        print ("\tWorking on Ex for l =", l)

        # RE-EVALUATE ENERGY
        level['E'] = QC.expectation(
            qctb.circuit.bind(energyop, x),
            template=template.format(**{f"x{i}": x[i] for i in range(len(x))}),
            filename=datafile,
        )

        # SAVE QPE FOR LATER
        if 'Ep' in level:   del level['Ep']

        # print ("\tWorking on Ep for l =", l)
        #
        # # RE-EVALUATE QPE
        # op = tools.hamiltonian(VQD.H, kpt)
        # level['Ep'] = qiskit.aqua.algorithms.IQPE(
        #     operator=op,
        #     state_in=qiskit.aqua.components.initial_states.Custom(
        #         VQD.H.M,
        #         circuit=qctb.circuit.bind(VQD.ansatz, x)
        #     ),
        #     num_iterations=13,
        #     expansion_mode='suzuki',
        #     expansion_order=2,
        #     quantum_instance=QC.qi,
        # ).compute_minimum_eigenvalue()['eigenvalue'].real

    rVQD.points.append(point)
    rVQD.save(savefile)

print ("fin")
