# https://github.com/marcobn/Quantum/blob/0b12d7ce39d589e0810c18f7b0eb183120c6ce38/BandTheory/plot_graphene.py
""" Generate band structure with VQD. """

n = None    # Reciprocal mode
n = [3]*2   # Real-space mode

run = True  # Collect data
run = False # Plot data

qpu = "svm" # Statevector simulator.
# qpu = "qvm" # Noiseless sampling.

import numpy as np
np.set_printoptions(3, suppress=True)
import qiskit

import qctb
import tightbinding





def plot_results_2D(VQD, N=None, plotx=None, plot0=None):
    """ Generate band structure x~K -> M -> Γ -> K~y.

    Requires VQD was constructed from run_band_2D.

    Parameters
    ----------
    VQD: qctb.vqd.VQD
        VQD object. VQD.H.D must be 1 for this function to work.
    plotx: vpython plotting object or list
         plots measured energies. If a list, length is VQD.H.M
    plot0: vpython plotting object or list
         plots analytical energies. If a list, length is VQD.H.M

    """
    # EXTEND PLOTTERS TO LISTS FOR CONVENIENCE
    if plot0 is not None:
        if not isinstance(plot0, list):
            plot0 = [plot0] * VQD.H.M
    if plotx is not None:
        if not isinstance(plotx, list):
            plotx = [plotx] * VQD.H.M

    # APPROXIMATE DEGENERACY POINT K
    n = 3 if VQD.n is None else VQD.n[0]
    N = 2**n
    x = N//3                # SMALLEST BINARY FRACTION LESS THAN 1/3
    y = N//3 + 1            # LARGEST BINARY FRACTION MORE THAN 1/3

    # SETUP PATH
    ps = []
    for i in range(x, N//2):      # FROM x -> M
        ps.append( [i, N-i] )           # ISOSURFACE OF KM BRANCH
    for i in range(N//2, 0, -1):  # FROM M -> Γ
        ps.append( [ i, i ] )           # ISOSURFACE OF MΓ BRANCH
    for i in range(0, y+1):         # FROM Γ -> y
        ps.append( [i, 2*i] )           # ISOSURFACE OF ΓK BRANCH
    ps = np.array(ps)
    # ptok = 2*np.pi * np.linalg.inv(VQD.H.R())

    # ANALYTICAL CURVES
    if plot0:
        res = 100                   # FINER SPACING IN PATH
        fineps = []
        for i in range(len(ps)-1):
            for λ in np.linspace(0,1,res, endpoint=False):
                fineps.append( λ*(ps[i+1]-ps[i]) + ps[i] )
        fineps.append(ps[-1])       # INCLUDE VERY LAST POINT
        fineps = np.array(fineps)

        for i, p in enumerate(fineps):
            kpt = VQD.H.kpt(p/N)
            E0 = VQD.H.E0(kpt)
            for l in range(VQD.H.M):        # PLOT EACH ENERGY
                plot0[l].plot(i/res, E0[l])

    # RUN EXPERIMENT
    for i, p in enumerate(ps):
        point = VQD.points[i]

        if plotx:
            for l in range(VQD.H.M):                # PLOT EACH ENERGY
                plotx[l].plot(i, point["levels"][l]["E"])









# CONSTRUCT FILENAME
filename = f"Graphene"
if n: filename += f".{n[0]}"
filename = f"json/{filename}.json"

if run:
    H = tightbinding.graphene.Graphene(0,1)
    if qpu == "svm":
        QC = qctb.ibmq.SVM
        OPT = qiskit.aqua.components.optimizers.SLSQP()
    elif qpu == "qvm":
        QC = qctb.ibmq.QVM
        OPT = qiskit.aqua.components.optimizers.COBYLA()
    VQD = qctb.vqd.VQD(H, QC, OPT, n=n, note="Graphene...")
else:
    VQD = qctb.vqd.load(filename)


import vpython
graph = vpython.graph(fast=False)
plot0 = [vpython.gcurve(graph=graph) for l in range(VQD.H.M)]
plotx =  vpython.gdots (graph=graph)

if run:
    qctb.experiment.run_band_2D(
        VQD,
        filename=filename,
        plotx=plotx,
        plot0=plot0
    )
else:
    plot_results_2D(VQD, plotx=plotx, plot0=plot0)

print ("fin")
