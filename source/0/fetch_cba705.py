# https://github.com/CS6620-S21/Asynchronous-quantum-job-submission-framework/blob/248f019e9bbc1971df9ffa5964a82037bd86a5f7/async_job/job-fetch-cronjob/fetch.py
  
#!/usr/bin/env python3
# Async Job
# Copyright(C) 2021 Team Async 
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Object Store Helper Library.

This is the job fetch cron-job that fetches all the pending jobs and checks if their result is available.
And moves the completed jobs to the Completed Bucket. 
"""

import os, logging, json
from qiskit import IBMQ
from async_job.api.object_store import ObjectStore
# from async_job.api.q_obj_encoder import QobjEncoder
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from datetime import datetime


COMPLETED_BUCKET = os.getenv("COMPLETED_BUCKET")
PENDING_BUCKET = os.getenv("PENDING_BUCKET")
BACKEND = os.getenv("BACKEND")

# Initialize ObjectStore
try:
    ob = ObjectStore()
except Exception as ex:
    logging.error("Error is -", ex)


def init_backend():
    """Returns the backend to work with in the cron job"""
    token=os.getenv('BACKEND_TOKEN')
    IBMQ.save_account(token, overwrite=True)
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    IBMQ.get_provider(group='open')
    try:
        backend = provider.get_backend(BACKEND)
    except QiskitBackendNotFoundError as er:
        logging.error("Qiskit Backend not found, please check backend in .env.")
        raise er
    return backend


def _generateResult(result_status, result_json):
    """Generate result json object based on result status."""
    if result_status == JobStatus.DONE:
        result_to_write = {
            "result": result_json,
            "timestamp": str(datetime.now())
        }
    else:
        result_to_write = {
            "result": str(result_status),
            "timestamp": str(datetime.now())
        }
    return result_to_write

def run_fetch_job():
    backend = init_backend()
    logging.info(f"Running with backend - {backend}")
    # Run for all pending objects 
    for pending_key in ob.get_all_objects(PENDING_BUCKET):
        # Process the pending object
        try:
            job_fetched = backend.retrieve_job(pending_key)
            # Check status in final state or not.
            job_status = job_fetched.status()
            if job_status in JOB_FINAL_STATES:
                result = job_fetched.result()
                result_dict = result.to_dict()
                result_json = json.dumps(result_dict, indent=4, sort_keys=True, default=str)
                result_json_object = _generateResult(result_status=job_status, result_json=result_json)
                try:
                    ob.put_object(job_body=result_json_object, file_name=pending_key, bucket_name=COMPLETED_BUCKET)
                    ob.delete_object(key=pending_key, bucket_name=PENDING_BUCKET)
                except Exception:
                    logging.error("Failed to write into completed bucket.")
            else:
                logging.info(f"Skipping job with id - {pending_key} as not in final state.")
        except Exception as ex:
            logging.error(f"Pending job {pending_key} couldn't be processed.", ex)
    
    logging.info(f"Completed running for all pending objects.")


if __name__ == "__main__":
    run_fetch_job()