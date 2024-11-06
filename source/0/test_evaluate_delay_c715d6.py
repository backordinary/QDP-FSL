# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/5d92c40dcfc68fab7922e806ab75e898bf3bda81/test/test_evaluate_delay.py
from qiskit import IBMQ

from experiments.utils import get_IBM_backend, pickle_load
from experiments.waiting_duration import DelayBenchmark, EvaluateDelay

def test_evaluate_delay():
    backend = get_IBM_backend("ibmq_rome")
    simulator = get_IBM_backend("ibmq_qasm_simulator")

    # load experiments data
    data_path = "/Users/Yasuhiro/Documents/aqua/gp/experiments/waiting_duration/job_id/2020-11-10_qc_name_is_ibmq_rome.pickle"
    exp_data = pickle_load(data_path)

    job_sim = simulator.retrieve_job(exp_data["simulator"]["job_id"])
    job_delay_op = [backend.retrieve_job(job_id) for job_id in exp_data["delay_op"]["job_id"]]
    job_delay_meas = [backend.retrieve_job(job_id) for job_id in exp_data["delay_meas"]["job_id"]]
    delay_duration_list = exp_data["delay_duration_list"]

    initial_layout=[0, 1, 2]

    eval_delay = EvaluateDelay(job_sim, job_delay_op, job_delay_meas, delay_duration_list, nseed=1, initial_layout=initial_layout)
    op_list, meas_list = eval_delay.evaluate()


    jsd_list_list, mean_list, sem_list = eval_delay.js_divergence(op_list)
    print(jsd_list_list, mean_list, sem_list)


if __name__ == "__main__":
    test_evaluate_delay()