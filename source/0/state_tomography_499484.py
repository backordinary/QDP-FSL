# https://github.com/dbucher97/ibmqx-bachelor/blob/bb95fae33083604f9b64ee06ca6deb64de5fb909/projects/state_tomography.py
from qiskit import QuantumProgram, Result
from qiskit.backends import discover_remote_backends
from IBMQuantumExperience import IBMQuantumExperience
from main import execute
import Qconfig
import qiskit.tools.qcvv.tomography as tomo
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import os, sys
from pprint import pprint
import pickle
import numpy as np

qp = QuantumProgram()

api = IBMQuantumExperience(Qconfig.APItoken, config=Qconfig.config)
discover_remote_backends(api)

n = 8

qr = qp.create_quantum_register("qr", n)
cr = qp.create_classical_register("cr", int(n/2))
qc = qp.create_circuit("tomo", [qr], [cr])

def pm():
    qc.h(qr)

def cat():
    qc.h(qr[0])
    for i in range(n-1):
        qc.cx(qr[i], qr[i+1])

def spt():
    pm()
    for i in range(n):
        qc.cz(qr[i], qr[(i+1)%n])


spt()
qc.barrier(qr)

# res = qp.execute(["tomo"], backend="local_qasm_simulator", shots=1)
#
# psi = res.get_data("tomo")["quantum_state"]
# print(psi)

# res = qp.run(qobj)

def split_qobj(qobj, M=5):
    c = 0
    l = []
    jl = []
    for circ in qobj.get("circuits"):
        if c==M:
            c=0
            jl += [[e for e in l]]
            l = []

        c+=1
        l += [{"name": circ.get("name"), "qasm": circ.get("compiled_circuit_qasm")}]
    jl += [[e for e in l]]
    return jl


def run_splitted_qasms(qasms, qobj, backend, shots=1024):
    path = "state_tomography/spt_%s"%backend
    os.system("rm %s*"%path)
    with open(path + ".pkl", 'wb') as f:
        pickle.dump(qobj, f, pickle.HIGHEST_PROTOCOL)
    jids = []
    for qasm in qasms:
        res = api.run_job(qasm, backend=backend, shots=shots)
        jid = res.get("id")
        os.system("echo '%s' >> state_tomography/spt_%s"%(jid, backend))
        jids += [jid]
        print(jid)
    return jids

def recover(s):
    with open(s+'.pkl', 'rb') as f:
        qobj = pickle.load(f)
    jids = open(s, "r").read().strip().split("\n")
    c = 1
    b = True
    xs = []
    for jid in jids:
        x = api.get_job(jid)
        print(c, x.get("status"))
        if x.get("status") != "COMPLETED":
            b = False
        xs += x["qasms"]
        c += 1
    if not b:
        sys.exit(1)
    return Result({"result": xs, "status": "DONE"}, qobj)

# tomo_set = tomo.state_tomography_set(list(range(4)))
# tomo_circuits = tomo.create_tomography_circuits(qp, 'tomo', qr, cr, tomo_set)
#
# res1 = qp.execute(tomo_circuits, backend="local_qiskit_simulator", shots=1024)

tomo_set = tomo.state_tomography_set(list(range(int(n/2))))

if sys.argv[1] == "run":
    backend = "ibmqx5"
    tomo_circuits = tomo.create_tomography_circuits(qp, 'tomo', qr, cr, tomo_set)
    qobj = qp.compile(tomo_circuits, backend=backend, shots=1024)

    qasms = split_qobj(qobj, M=1)
    run_splitted_qasms(qasms, qobj, backend)
    sys.exit(0)
elif sys.argv[1] == "simulate":
    tomo_circuits = tomo.create_tomography_circuits(qp, 'tomo', qr, cr, tomo_set)
    qobj = qp.compile(tomo_circuits, backend="local_qiskit_simulator", shots=1024)

    res = qp.run(qobj)
else:
    res = recover(sys.argv[1])
    # pprint(res1._result)
    # pprint(res._result)
    # print(res.get_counts("tomo_meas_X(0)X(1)X(2)X(3)"))
tomo_data = tomo.tomography_data(res, "tomo", tomo_set)
rho_fit = tomo.fit_tomography_data(tomo_data)

np.save("state_tomography/rho.npy", rho_fit)
ev = eigvals(rho_fit)

plt.bar(range(len(ev)), ev.real)
plt.show()
