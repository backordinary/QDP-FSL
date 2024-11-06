# https://github.com/ccicconetti/qiskit-tests/blob/8e362a1d041b071b64b9630201e0bb56246c735a/utils/job_status.py
#!/usr/bin/env python3

import argparse
import json

from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus
from tutorial.utils import splitProjectInfo


def print_job_info(job):
    print(
        f"name {job.name()} ({job.creation_date()}) jobid {job.job_id()} status {job.status()}"
    )


parser = argparse.ArgumentParser(
    "Query job", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--project", type=str, default="", help="Indicate a specific hub,group,project"
)
parser.add_argument(
    "--backend",
    type=str,
    default="",
    help="IBMQ backend running the experiment",
)
parser.add_argument(
    "--list", action="store_true", default=False, help="List all the jobs"
)
parser.add_argument("--save", type=str, default="", help="Save the result to file")
parser.add_argument("--jobid", type=str, default="", help="Job ID")
args = parser.parse_args()

assert args.list or args.jobid != ""
assert args.backend != ""

provider = IBMQ.load_account()
if args.project:
    (hub, group, project) = splitProjectInfo(args.project)
    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
backend = provider.get_backend(args.backend)

if args.list:
    for job in backend.jobs():
        print_job_info(job)
else:
    job = backend.retrieve_job(args.jobid)
    print_job_info(job)
    if job.status() == JobStatus.DONE and args.save != "":
        with open(args.save, "w") as outfile:
            json.dump(job.result().to_dict(), outfile, default=str)
