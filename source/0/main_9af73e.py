# https://github.com/dbucher97/ibmqx-bachelor/blob/bb95fae33083604f9b64ee06ca6deb64de5fb909/projects/main.py
import Qconfig
import sys
import json
import os
import time
import datetime
import cursor
from multiprocessing.pool import ThreadPool
from IBMQuantumExperience import IBMQuantumExperience
from qiskit import QuantumProgram, QuantumCircuit
from qiskit.backends import discover_remote_backends
from qiskit.tools.visualization import plot_histogram, plot_circuit
# from qiskit.tools.file_io import file_datestr, save_result_to_file, load_result_from_file
from file_io import file_datestr, save_result_to_file, load_result_from_file
from qbit_mapping import unscramble_counts

BASE_DIR = "/home/david/bachelor/ibmqx/projects/"
os.chdir(BASE_DIR)

meta_results = []
api = None

col_list = ["#255F85", "#E9724C", "#358229", "#F22B32", "#BBAB0B"]
# col_list = ["#235789", "#C1292E", "#5BAA1D", "#E58B19", "#161925"]
colors = {"exp": col_list[0], "exp0": col_list[0], "exp1": col_list[2], "exp2": col_list[3], "exp3":
          col_list[4], "sim": col_list[1]}

def is_simulation(backend):
    if backend in ["ibmqx2", "ibmqx4", "ibmqx5"]:
        return False
    else:
        return True


def get_name():
    return sys.argv[0][:-3]

def get_api():
    global api
    return api


def setup_circuit(qp, num_reg, classical=None, additional_registers=None, name=None):
    # Adds circiut to existing program.
    # Automatic Naming
    if name == None:
        name = get_name()
    if classical == None:
        classical=num_reg
    c = 0
    post = ''
    ppost = '%d' % len(qp.get_circuit_names()) if len(qp.get_circuit_names()) > 0 else ''
    while name+post in qp.get_circuit_names():
        c += 1
        post = '_%d' % c
    qr = qp.create_quantum_register('qr'+ppost, num_reg)
    cr = qp.create_classical_register('cr'+ppost, classical)
    qrs = []
    crs = []
    if additional_registers:
        if "qr" in additional_registers:
            for reg_name, reg_len in additional_registers["qr"].items():
                qrs += [qp.create_quantum_register(reg_name, reg_len)]
        if "cr" in additional_registers:
            for reg_name, reg_len in additional_registers["cr"].items():
                crs += [qp.create_classical_register(reg_name, reg_len)]
    qc = qp.create_circuit(name+post, [qr]+qrs, [cr]+crs)
    if qrs:
        qr = [qr] + qrs
    if crs:
        cr = [cr] + crs
    return qc, qr, cr

def register():
    global api
    if not api:
        api = IBMQuantumExperience(token=Qconfig.APItoken, config=Qconfig.config)
        backends = discover_remote_backends(api)
        print("Logged in. Possible backends: " + ", ".join(backends))


def setup_only(login=False):
    # Returns Quantum Program, Quantum Circuit.
    qp = QuantumProgram()
    if login and not (len(sys.argv) > 1 and (sys.argv[1] == "load" or sys.argv[1] == "simulate")):
        register()
            # deprecated
            # qp.set_api(Qconfig.APItoken, Qconfig.config['url'])
    return qp

def setup(num_reg, additional_registers=None, name=None, classical=None, login=False):
    # Returns Quantum Program, Quantum Circuit.
    qp = setup_only(login)
    qc, qr, cr = setup_circuit(qp, num_reg, name=name, classical=classical,
                               additional_registers=additional_registers)
    return qp, qc, qr, cr


def execute_on_device(qp, circuits, backend="local_qiskit_simulator", shots=1024, timeout=None,
                      wait=10, max_credits=3, config=None, coupling_map=None, basis_gates=None,
                      hpc=None, initial_layout=None):
    if backend not in ["ibmqx5", "ibmqx4", "ibmqx2"]:
        result = qp.execute(circuits, backend=backend, shots=shots, timeout=timeout, wait=wait,
               max_credits=max_credits, config=config, coupling_map=coupling_map,
                            basis_gates=basis_gates, hpc=hpc, initial_layout=initial_layout)
    else:
        pool = ThreadPool(processes=1)
        if not timeout:
            timeout=max(400, 400*api.backend_status(backend).get("pending_jobs"))
        async_result = pool.apply_async(qp.execute, (circuits,),
                                        {"backend": backend, "shots": shots, "timeout": timeout,
                                         "wait": wait, "max_credits": max_credits, "config": config,
                                         "coupling_map": coupling_map, "basis_gates": basis_gates,
                                         "hpc": hpc, "initial_layout": initial_layout})
        c = 0
        try:
            cursor.hide()
            st = time.time()
            while not async_result.ready():
                if c%50 == 0:
                    device_info = api.backend_status(backend)
                    if "busy" not in device_info:
                        device_info["busy"] = False
                print("Running on %s %s\tTime %s/%d\tDevice is %sbusy. %d pending jobs."
                      %(backend, "/-\\|"[c%4], str(int(time.time()-st)).rjust(len(str(timeout)), " "),
                        timeout, "" if device_info["busy"] else "not ", device_info["pending_jobs"]),
                      end="\r")
                c += 1
                time.sleep(0.2)
            cursor.show()
            print("")
        except KeyboardInterrupt:
            cursor.show()
            print("")
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        result = async_result.get()
    return result




def execute(qp, circuits=None, backend="local_qiskit_simulator", shots=1024, sav=1, meta=None,
            unscramble=True, max_credits=3, config=None, coupling_map=None, basis_gates=None,
            hpc=None, initial_layout=None):
    # Executes specified circuits. If none specified, executes all.
    if len(sys.argv) > 1 and (sys.argv[1] == "load" and backend in ["ibmqx2", "ibmqx4", "ibmqx5"]):
        h = 1
        try:
            h = int(sys.argv[2])
        except IndexError:
            pass
        except ValueError:
            pass
        result = load_result(meta=meta, h=h)
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "simulate":
            backend = "local_qiskit_simulator"
        if circuits == None:
            circuits = list(qp.get_circuit_names())
        result = None
        print("\nEXECUTING " + ", ".join(circuits)+ " ON " + backend+ "\n")
        while result == None or result.get_status() == "ERROR":
            if not result == None:
                jid = result.get_job_id()
                print(jid, "failed, retry ...")
                add_failed_jobids([jid], name=result._qobj["circuits"][0]["name"])
            result = execute_on_device(qp, circuits, backend=backend,
                                shots=shots, wait=10, max_credits=max_credits,
                                config=config, coupling_map=coupling_map, basis_gates=basis_gates,
                                hpc=hpc, initial_layout=initial_layout)
        if sav == 2 or (sav == 1 and not is_simulation(backend)):
            save_result(result, meta=meta)
    if unscramble and backend == "ibmqx5":
        for d in result._result["result"]:
            dat = datetime.datetime.strptime(d["data"]["date"][:-5], "%Y-%m-%dT%H:%M:%S")
            if dat < datetime.datetime(2018, 4, 15):
                d["data"]["counts"] = unscramble_counts(d["data"]["counts"])
    return result

def add_failed_jobids(jobids, name=None):
    if name == None:
        name = get_name()
    make_dirs(name, False)
    path = os.path.join(name, "failed")
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, "job_ids.txt"), "a") as f:
        for jid in jobids:
            f.write(jid+"\t"+'{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now())+"\n")


def make_dirs(name, simulation):
    path = name+"/"
    try:
        os.mkdir(name)
    except FileExistsError:
        pass
    if simulation:
        path += "simulations/"
        try:
            os.mkdir(name+"/simulations")
        except FileExistsError:
            pass
    return path


def save_result(result, name=None, meta=None):
    # saves the data as json
    if not name:
        name = get_name()
    backend = result._qobj.get("config").get("backend")
    path = make_dirs(name, is_simulation(backend))
    save_result_to_file(result, file_datestr(path, name), metadata=meta)


def load_result(name=None, filename=None, meta=None, h=1, simulations=False):
    if filename == None:
        if not name:
            name = get_name()
        if not os.path.isdir(name):
            print("project '%s' not started yet" % name)
            sys.exit(1)
        d = []
        if not simulations:
            d = sorted(os.listdir(name), reverse=True)
            d = list(map(lambda s: os.path.join(name, s), d))
            d = [e for e in d if os.path.isfile(e)]
        if not d:
            if not simulations:
                print("no saved experiments for '%s', looking for simulations" % name)
            name = os.path.join(name, "simulations")
            d = sorted(os.listdir(name), reverse=True)
            d = list(map(lambda s: os.path.join(name, s), d))
            d = [e for e in d if os.path.isfile(e)]
        if not d:
            print("no files found for '%s'" % name)
            sys.exit(1)
        cur = ""
        c = 0
        for e in d:
            c += 1
            cur = e
            if meta:
                x = load_result_from_file(e)
                if x[1] != meta:
                    c -= 1
            if c == h:
                break
        filename = cur
    result, meta = load_result_from_file(filename)
    jid = result.get_job_id()
    print("loaded " + meta, "; " + jid)
    if os.path.isfile("confs/bin"):
        with open("confs/bin", "a") as f:
            f.write(jid+" ")
    add_meta_for_result(result, meta)
    return result


def add_meta_for_result(result, meta):
    global meta_results
    if meta:
        meta_results += [(meta, result)]


def get_meta_from_result(result):
    global meta_results
    for m, r in meta_results:
        if result == r:
            return m

def visualize(result, circuits=None, ntk=False):
    # Visualizes specified curcuit results as histogram.
    if circuits == None:
        circuits = result.get_names()
    for name in circuits:
        plot_histogram(result.get_counts(name), ntk)

def get_compiled_qasm(qp, name=None, backend="ibmqx5"):
    if name==None:
        name=get_name()
    global api
    if backend in ["ibmqx2", "ibmqx4", "ibmqx5"] and not api:
        api = IBMQuantumExperience(token=Qconfig.APItoken, config=Qconfig.config)
        backends = discover_remote_backends(api)
        print("Logged in. Possible backends: " + ", ".join(backends))
    qobj = qp.compile([name], backend=backend)
    return qp.get_compiled_qasm(qobj, name)

def count_gates(qasm):
    lines = qasm.split("\n")
    d = {"total": 0, "single": 0,  "controlled": 0}
    no_gates = ["OPENQASM", "creg", "qreg", "include", "barrier", "measure"]
    for line in lines:
        if sum([line.startswith(g) for g in no_gates]) == 0 and line != "":
            if line[0] == "c":
                d["controlled"] += 1
            else:
                d["single"] += 1
            d["total"] += 1
    return d

def get_calibration(jid):
    global api
    if not api:
        register()
    j = api.get_job(jid)
    return j["calibration"]

def tex_calibration(jid):
    calibration = get_calibration(jid)
    dat = calibration["lastUpdateDate"]
    d = {}
    for qubit in calibration["qubits"]:
        name = qubit.get("name")
        if name:
            dd = {}
            dd["T1"] = qubit.get("T1").get("value")
            dd["T2"] = qubit.get("T2").get("value")
            dd["gate_error"] = qubit.get("gateError").get("value")
            dd["readout_error"] = qubit.get("readoutError").get("value")
            d[name] = dd
    import tabulate
    kv = {"T1": "$T_1\,[\mu s]$", "T2": "$T_2\,[\mu s]$", "readout_error": "$\epsilon_r$", "gate_error": "$\epsilon_g$"}
    s = tabulate.tabulate([["Qubits", *list(map(lambda s: "$"+s[0]+"_"+s[1:]+"$", d.keys()))],
        *[[v, *list(map(lambda x: round(x[k], 3), d.values()))] for k, v in kv.items()]], tablefmt="latex_raw")
    os.system("echo '%s' > confs/%s.tex"%(s, dat))
