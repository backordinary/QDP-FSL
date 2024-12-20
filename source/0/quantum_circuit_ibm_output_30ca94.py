# https://github.com/Quantum-Creative-Group/quantum_nodes/blob/453031da5515d3507f885624d005e091f619d2e5/quantum_nodes/nodes/quantum_output/quantum_circuit_ibm_output.py
import bpy
from bpy.types import Node
from bpy.props import BoolProperty, IntProperty, EnumProperty

import time
from qiskit import IBMQ
from qiskit.compiler import transpile, assemble
from animation_nodes.base_types import AnimationNode
from qiskit.providers.jobstatus import JOB_FINAL_STATES


class Provider():
    """Used as attribute in the node IBM Output."""

    def __init__(self):
        self.provider = None

    def get_provider(self):
        if self.provider is None:
            self.provider = IBMQ.providers()[0]
        return self.provider


class QuantumCircuitIBMOutputStateNode(Node, AnimationNode):
    """Output of the circuit sent to IBM computers."""

    bl_idname = "an_QuantumCircuitIBMOutputStateNode"
    bl_label = "Quantum Circuit IBM Output"
    bl_width_default = 210
    errorHandlingType = "EXCEPTION"
    provider = Provider()

    initialized: BoolProperty(
        name="Initialized",  # noqa F821
        description="If the node has been initialized",
        default=False,
    )

    remaining_jobs: IntProperty(
        name="Remaining jobs",
        description="The number of remaining jobs for a backend",
        default=0,
        soft_min=0,
        min=0,
    )

    def item_callback(self, _context):
        if self.initialized:
            return [(sys.name(), sys.name(), "number of qubits: " + str(sys.configuration().n_qubits))
                    for sys in self._provider.get_provider().backends()]
        else:
            return []

    backendMenu: EnumProperty(
        name="Backend",  # noqa F821
        description="Choose a system",
        items=item_callback,
        update=AnimationNode.refresh,
    )

    def setup(self):                            # disables auto-load at creation
        node_tree = bpy.context.space_data.edit_tree
        node_tree.autoExecution.enabled = False
        if not self.initialized:
            if IBMQ.active_account() is None:   # test if the IBMQ account is already loaded
                try:
                    IBMQ.load_account()
                    self.initialized = True
                except BaseException:

                    self.raiseErrorMessage("You are not connected to any IBM account.\
                        Please enter your token in the Quantum Node panel or check your internet connection.")
                    bpy.ops.wm.call_panel(name="AN_PT_InsertNodeUI")

    def create(self):
        self.newInput("Quantum Circuit", "Quantum Circuit", "quantum_circuit")
        self.newOutput("Generic", "Output State", "output_state")

    def draw(self, layout):
        layout.prop(self, "backendMenu")
        layout.label(text="Number of jobs remaining: " + str(self.remaining_jobs), icon="INFO")
        self.invokeFunction(layout, "executeTree", text="Send")

    def executeTree(self):
        node_tree = bpy.context.space_data.edit_tree
        node_tree.execute()

    def execute(self, quantum_circuit):
        if self.initialized:
            backend = self._provider.get_provider().get_backend(self.backendMenu)
            self.remaining_jobs = backend.remaining_jobs_count()
            if backend.status().operational is False:
                self.raiseErrorMessage("This system is offline for now")
            if quantum_circuit.num_qubits > backend.configuration().n_qubits:
                self.raiseErrorMessage("This system doesn't compute enough\
                    qubits: " + str(backend.configuration().n_qubits))
            # Prepare the job
            qobj = assemble(transpile(quantum_circuit, backend=backend), backend=backend)
            job = backend.run(qobj)
            # Display the progress
            start_time = time.time()
            job_status = job.status()
            while job_status not in JOB_FINAL_STATES:
                print(f'Status @ {time.time()-start_time:0.0f} s: {job_status.name},'
                      f' est. queue position: {job.queue_position()}')
                job_status = job.status()
            # Handle the result
            result = job.result()
            return result.get_counts()
        else:
            return
