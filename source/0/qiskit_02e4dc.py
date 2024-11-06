# https://github.com/CQCL/ethz-hackathon22/blob/3b78e0a02aa9c7b71538bbe763e598f4fa147a34/benchmarking/compilers/qiskit.py
from .compiler import Compiler
import qiskit
from pytket import Circuit
from qiskit.transpiler import PassManagerConfig, CouplingMap
from qiskit.transpiler.preset_passmanagers import level_3_pass_manager, level_0_pass_manager
from pytket.passes import RebaseQuil
from utils import load_coupling, post_process
from pytket.extensions.qiskit import tk_to_qiskit, qiskit_to_tk
import time

class QiskitCompiler(Compiler):
    compiler_id = "qiskit"

    def __init__(self, backend_name, optimising=True):

        self.version = qiskit.__version__

        self.coupling_map = load_coupling(backend_name)

        self.optimising = optimising
        
        pmc = PassManagerConfig(coupling_map=CouplingMap(self.coupling_map), basis_gates=['u1', 'u3', 'cx', 'u3'])
        if self.optimising:
            self.pm = level_3_pass_manager(pmc)
        else:
            self.pm = level_0_pass_manager(pmc)

        self.backend_name = backend_name
        if self.optimising:
            self.name = "optimising qiskit"
        else:
            self.name = "non-optimising qiskit"

    def compile(self, orig_circ: Circuit) -> Circuit:

        circ = orig_circ.copy()
        qiskit_circ = tk_to_qiskit(circ)

        start_time = time.time()
        qiskit_circ = self.pm.run(qiskit_circ)
        time_elapsed = time.time() - start_time

        circ = qiskit_to_tk(qiskit_circ)
        RebaseQuil().apply(circ)
        if self.optimising:
            circ = post_process(circ)

        return circ, time_elapsed

exports_compilers = [QiskitCompiler]
