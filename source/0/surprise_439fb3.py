# https://github.com/Maalshekto/airflow_dags/blob/7a552f9e04786b9d3cb09d1078d51725b877ddae/surprise.py
import airflow
import time
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow import DAG

from swiftclient.service import SwiftService
from swiftclient.service import SwiftUploadObject
import os
import subprocess

from qiskit import IBMQ, assemble, transpile, execute, Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.visualization import plot_histogram

import pandas as pd
import matplotlib.pyplot as plt

NB_SHOTS = 4000 # Equal to the number of shots of the real quantum backend.
BUCKET_QUANTUM = "swift_quantum"
PERFECT_QUANTUM_PLOT_JPG = "histogram_perfect.jpg"
NOISY_QUANTUM_PLOT_JPG = "histogram_noisy.jpg"
FINAL_RESULTS_PLOT_JPG = "final_results.jpg"

IBMQ.save_account('76416dc2d7a314e56fb9fafd05a24607c8060643a7a3265055655f27e48811d5692d4567c6a2fa82ce69490b237465164c4a9653a13594895eff039f27c6780d')
provider = IBMQ.load_account()
#qx = random_circuit(5, 4, measure=True)
#qr=QuantumRegister(2)
#cr=ClassicalRegister(2)
#qx=QuantumCircuit(qr,cr)
#qx.h(qr[0])
#qx.cx(qr[0],qr[1])
#qx.measure(qr,cr)

qr=QuantumRegister(1)
cr=ClassicalRegister(1)
qx=QuantumCircuit(qr,cr)
qx.initialize(0,0)
qx.h(qr[0])
qx.h(qr[0])
qx.measure(qr,cr)

def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""
    pipe = subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True, encoding='utf8')
    output = pipe.communicate()[0]
    env = dict((line.split("=", 1) for line in output.splitlines()))
    os.environ.update(env)

def _real_quantum_backend(ti): 
  backend = provider.backend.ibm_oslo
  transpiled = transpile(qx, backend=backend, optimization_level=0)
  job = backend.run(transpiled)
  retrieved_job = backend.retrieve_job(job.job_id())
  retrieved_job.wait_for_final_state()
  result = retrieved_job.result()
  ti.xcom_push(key='counts_experiment', value=result.get_counts(qx))

def _fake_quantum_backend(ti): 
  time.sleep(10)
  ti.xcom_push(key='counts_experiment', value={'01': 79, '10': 70, '00': 1959, '11': 1892})

def _simulator_perfect_quantum_backend(ti):
  backend = Aer.get_backend('aer_simulator')
  transpiled = transpile(qx, backend=backend, optimization_level=0)
  job = backend.run(transpiled, shots = NB_SHOTS)
  result = job.result()
  counts = result.get_counts(qx)
  print(counts)
  plot_histogram(counts, filename = f"/tmp/{PERFECT_QUANTUM_PLOT_JPG}")
  
  # Set OpenStack connection variables
  shell_source("/app/openrc/openrc.sh")
  print(f"OS_TENANT_NAME {os.getenv('OS_TENANT_NAME')}")
  # retrieve from Swift container
  options = {
    "auth_version" : os.getenv('OS_IDENTITY_API_VERSION'),
    "os_username" : os.getenv('OS_USERNAME'),
    "os_password" : os.getenv('OS_PASSWORD'),
    "os_tenant_name" : os.getenv('OS_TENANT_NAME'),
    "os_auth_url" :  os.getenv('OS_AUTH_URL'),
    "os_region_name" : os.getenv('OS_REGION_NAME'),
  }
  
  with SwiftService(options=options) as swift:
    for up_res in swift.upload(BUCKET_QUANTUM, [
      SwiftUploadObject(f"/tmp/{PERFECT_QUANTUM_PLOT_JPG}", object_name=PERFECT_QUANTUM_PLOT_JPG)]):
      if up_res['success']:
        print("'%s' uploaded" % PERFECT_QUANTUM_PLOT_JPG)
      else:
        print("'%s' upload failed" % PERFECT_QUANTUM_PLOT_JPG) 
  ti.xcom_push(key='counts_experiment', value=counts)

def _simulator_noisy_quantum_backend(ti):
  backend = provider.backend.ibm_oslo
  noise_model = NoiseModel.from_backend(backend)
  # Get coupling map from backend
  coupling_map = backend.configuration().coupling_map

  # Get basis gates from noise model
  basis_gates = noise_model.basis_gates
  
  result = execute(qx, Aer.get_backend('qasm_simulator'),
                 optimization_level=0,
                 coupling_map=coupling_map,
                 basis_gates=basis_gates,
                 noise_model=noise_model,
                 shots = NB_SHOTS).result()
  counts = result.get_counts(qx)
  print(counts)
  plot_histogram(counts, filename = f"/tmp/{NOISY_QUANTUM_PLOT_JPG}")
  
  # Set OpenStack connection variables
  shell_source("/app/openrc/openrc.sh")
  print(f"OS_TENANT_NAME {os.getenv('OS_TENANT_NAME')}")
  # retrieve from Swift container
  options = {
    "auth_version" : os.getenv('OS_IDENTITY_API_VERSION'),
    "os_username" : os.getenv('OS_USERNAME'),
    "os_password" : os.getenv('OS_PASSWORD'),
    "os_tenant_name" : os.getenv('OS_TENANT_NAME'),
    "os_auth_url" :  os.getenv('OS_AUTH_URL'),
    "os_region_name" : os.getenv('OS_REGION_NAME'),
  }
  
  with SwiftService(options=options) as swift:
    for up_res in swift.upload(BUCKET_QUANTUM, [
      SwiftUploadObject(f"/tmp/{NOISY_QUANTUM_PLOT_JPG}", object_name=NOISY_QUANTUM_PLOT_JPG)]):
      if up_res['success']:
        print("'%s' uploaded" % NOISY_QUANTUM_PLOT_JPG)
      else:
        print("'%s' upload failed" % NOISY_QUANTUM_PLOT_JPG) 
  ti.xcom_push(key='counts_experiment', value=counts)

def _print_result(ti):
  counts_experiment = ti.xcom_pull(key='counts_experiment', task_ids=['real_quantum_backend', 'simulator_perfect_quantum_backend', 'simulator_noisy_quantum_backend'])
  print(f'Results : {counts_experiment}')
  df = pd.DataFrame(counts_experiment)
  df = df.reindex(sorted(df.columns), axis=1)
  df = df.fillna(0)
  df.index = [ 'Real backend', 'Perfect simulated backend', 'Noisy simulated backend' ]
  fig = df.plot(kind='barh', subplots=True, figsize=(16,10))[0].get_figure() 
  plt.tight_layout()
  fig.savefig(f"/tmp/{FINAL_RESULTS_PLOT_JPG}")

  # Set OpenStack connection variables
  shell_source("/app/openrc/openrc.sh")
  print(f"OS_TENANT_NAME {os.getenv('OS_TENANT_NAME')}")
  
  # retrieve from Swift container
  options = {
    "auth_version" : os.getenv('OS_IDENTITY_API_VERSION'),
    "os_username" : os.getenv('OS_USERNAME'),
    "os_password" : os.getenv('OS_PASSWORD'),
    "os_tenant_name" : os.getenv('OS_TENANT_NAME'),
    "os_auth_url" :  os.getenv('OS_AUTH_URL'),
    "os_region_name" : os.getenv('OS_REGION_NAME'),
  }
  
  with SwiftService(options=options) as swift:
    for up_res in swift.upload(BUCKET_QUANTUM, [
      SwiftUploadObject(f"/tmp/{FINAL_RESULTS_PLOT_JPG}", object_name=FINAL_RESULTS_PLOT_JPG)]):
      if up_res['success']:
        print("'%s' uploaded" % FINAL_RESULTS_PLOT_JPG)
      else:
        print("'%s' upload failed" % FINAL_RESULTS_PLOT_JPG) 
 

dag = DAG (
    dag_id="surprise",
    start_date=airflow.utils.dates.days_ago(14),
    schedule_interval=None,
)

create_swift_object_storage = BashOperator(
    task_id = "create_swift_object_storage",
    bash_command=f"source /app/openrc/openrc.sh; swift post {BUCKET_QUANTUM};",
    do_xcom_push=False,
    dag=dag,
)

Q1 = PythonOperator( 
    task_id="real_quantum_backend",
    python_callable=_real_quantum_backend, 
    dag=dag,
)

Q2 = PythonOperator(
    task_id="simulator_perfect_quantum_backend",
    python_callable=_simulator_perfect_quantum_backend, 
    dag=dag,
)

Q3 = PythonOperator(
    task_id="simulator_noisy_quantum_backend",
    python_callable=_simulator_noisy_quantum_backend, 
    dag=dag,
)

res= PythonOperator(
    task_id="print_result",
    python_callable=_print_result, 
    dag=dag,
)

create_swift_object_storage >> [Q1, Q2, Q3] >> res
