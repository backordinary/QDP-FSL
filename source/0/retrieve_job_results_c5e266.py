# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/retrieve_job_results.py
from qiskit import IBMQ
from qiskit.tools.visualization import plot_histogram
import sys


def get_job_status(backend, job_id):
    backend = IBMQ.get_backend(backend)
    try:
        print("Backend {0} is operational? {1}".format(
            backend.name(),
            backend.status()['operational']))
        print("Backend was last updated in {0}".format(
            backend.properties()['last_update_date']))
        print("Backend has {0} pending jobs".format(
            backend.status()['pending_jobs']))
    except:
        print("Maybe some errors in the properties of backend")
    ibmq_job = backend.retrieve_job(job_id)
    status = ibmq_job.status()
    print(status.name)
    print(ibmq_job.creation_date())
    if (status.name == 'DONE'):
        print("EUREKA")
        result = ibmq_job.result()
        counts = result.get_counts()
        print("Result is: {0}".format(counts))
        plot_histogram(counts)
    else:
        print("... Work in progress ...")
        print("Queue position = {0}".format(ibmq_job.queue_position()))
        print("Error message = {0}".format(ibmq_job.error_message()))


IBMQ.load_accounts()
print("Account(s) loaded")

if (len(sys.argv) > 2):
    get_job_status(sys.argv[1], sys.argv[2])
else:  # All jobs from file
    print("Accessing all jobs from file")
    with open("jobs.txt") as f:
        for line in f:
            print("********")
            line = line.strip('\n').strip('\r')
            # If the job is not commented out for whatever reason
            if (not (line.startswith("#"))):
                vals = line.split(';')
                print("Job name is {0}".format(vals[0]))
                get_job_status(vals[1], vals[2])
