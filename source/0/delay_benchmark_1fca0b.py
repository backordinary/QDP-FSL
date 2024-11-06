# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/5d92c40dcfc68fab7922e806ab75e898bf3bda81/waiting_duration/delay_benchmark.py
from typing import List

from qiskit import QuantumCircuit, execute
from qiskit.compiler import sequence, transpile
from qiskit.circuit import Measure

from .insert_delay import InsertDelay
from .evaluate_delay import EvaluateDelay


class DelayBenchmark:
    def __init__(self, benchmark_qc: QuantumCircuit):
        self.benchmark_qc = benchmark_qc.remove_final_measurements(inplace=False)

    def compose(
        self,
        delay_duration_list: List[int],
    ):

        self.delay_before_list = []
        self.delay_after_list = []
        self.circ_sim = self.benchmark_qc.copy(name="simulator")
        self.circ_sim.measure_all()
        for delay_duration in delay_duration_list:
            delayed_circ = InsertDelay(delay_duration, self.benchmark_qc)

            delayed_before = delayed_circ.before_operation()
            self.delay_before_list.append(delayed_before)

            delayed_after = delayed_circ.after_operation()
            self.delay_after_list.append(delayed_after)

    def run(self, backend, simulator, initial_layout, nseed=1):
        job_sim = execute(self.circ_sim, backend=simulator, shots=8192)

        job_delay_before_list = []
        job_delay_after_list = []
        for seed in range(nseed):
            # transpiled_delay_before_list = transpile(self.delay_before_list, backend=backend, initial_layout = initial_layout, basis_gates=['u1', 'u2', 'u3', 'cx'])
            job_delay_before = execute(
                self.delay_before_list,
                backend=backend,
                shots=8192,
                # basis_gates=["u1", "u2", "u3", "cx"],
                initial_layout=initial_layout,
                schedule_circuit=True,
            )
            job_delay_before_list.append(job_delay_before)

            # transpiled_delay_after_list = transpile(self.delay_after_list, backend=backend, basis_gates=['u1', 'u2', 'u3', 'cx'])
            job_delay_after = execute(
                self.delay_after_list,
                backend=backend,
                shots=8192,
                # basis_gates=["u1", "u2", "u3", "cx"],
                initial_layout=initial_layout,
                schedule_circuit=True,
            )
            job_delay_after_list.append(job_delay_after)

        if nseed == 1:
            return job_sim, job_delay_before_list[0], job_delay_after_list[0]
        return job_sim, job_delay_before_list, job_delay_after_list

    def result(job_sim, job_delay_before, job_delay_after, delay_duration_list):
        return EvaluateDelay(
            job_sim, job_delay_before, job_delay_after, delay_duration_list
        )
