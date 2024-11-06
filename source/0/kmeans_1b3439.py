# https://github.com/vietzd/qc-cloud-challenges/blob/61d00e6a11b3d48ccc1d7219b1d6fcf382e29393/kmeans/faas/kmeans.py
import numpy as np
import math
from math import pi
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from cloudant.client import Cloudant


def get_theta(d):
    x = d[0]
    y = d[1]
    theta = 2*math.acos((x+y)/2.0)
    return theta


def get_Distance(x,y):
    theta_1 = get_theta(x)
    theta_2 = get_theta(y)

    # TODO phi

    qr = QuantumRegister(3, name="qr")
    cr = ClassicalRegister(3, name="cr")
    qc = QuantumCircuit(qr, cr, name="k_means")

    qc.h(qr[0])
    qc.h(qr[1]) # vermutlich falsch
    qc.h(qr[2]) # vermutlich falsch
    qc.u(theta_1, pi, pi, qr[1])
    qc.u(theta_2, pi, pi, qr[2])
    qc.cswap(qr[0], qr[1], qr[2])
    qc.h(qr[0])
    qc.measure(qr[0], cr[0])

    qc.reset(qr)
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend=backend, shots=1024)
    data = job.result().get_counts(qc)

    if len(data)==1:
        return 0.0
    else:
        return data['001']/1024.0


def get_data():
    client = Cloudant.iam(CLOUDANT_ACCOUNT_NAME, CLOUDANT_API_KEY, connect=True)
    kmeans_train = client['kmeans_train']

    ar = []
    for doc in kmeans_train:
        ar.append([doc['x'], doc['y']])
    return np.array(ar)


def randomly_initialize_centers(points,k):
    return points[np.random.randint(points.shape[0],size=k),:]

# Get a list of eachs point's nearest centroid
# Iterate over all points and check the distance to each centroid
def find_nearest_neighbour(points,centroids):
    n = len(points)
    k = centroids.shape[0]
    centers = np.zeros(n)
    for i in range(n):
        min_dis = 10000
        ind = 0
        for j in range(k):
            temp_dis = get_Distance(points[i,:],centroids[j,:])
            if temp_dis < min_dis:
                min_dis = temp_dis
                ind = j
        centers[i] = ind
    return centers


# The centroids for all clusters are adjusted by calculating the mean of the associated points
def find_centroids(points,centers):
    n = len(points)
    k = int(np.max(centers))+1
    centroids = np.zeros([k,2])
    for i in range(k):
        centroids[i,:] = np.average(points[centers==i])
    return centroids


def saveToDB(centroid, db_name):
    print("Save to DB", db_name, "...")
    client = Cloudant.iam(CLOUDANT_ACCOUNT_NAME, CLOUDANT_API_KEY, connect=True)
    db_centroids = client[db_name]
    doc = db_centroids["Black"]
    doc['x'] = centroid[0][0]
    doc['y'] = centroid[0][1]
    doc.save()
    doc = db_centroids["Blue"]
    doc['x'] = centroid[1][0]
    doc['y'] = centroid[1][1]
    doc.save()
    doc = db_centroids["Green"]
    doc['x'] = centroid[2][0]
    doc['y'] = centroid[2][1]
    doc.save()
    print("...finished")
    client.disconnect()


def main(dict):
    global CLOUDANT_ACCOUNT_NAME
    CLOUDANT_ACCOUNT_NAME = dict['CLOUDANT_ACCOUNT_NAME']
    global CLOUDANT_API_KEY
    CLOUDANT_API_KEY = dict['CLOUDANT_API_KEY']
    k = dict['k']
    it = dict['iterations']
    print("Try to find", k, "clusters with", it, "iterations")
    print("Get Data from Database...")
    points = get_data()
    print("Found ", len(points), "datapoints")
    print("Randomize initial centers...")
    centroids = randomly_initialize_centers(points,k)
    print("Centroid after iteration 0:", centroids.tolist())
    for i in range(it):
        centers = find_nearest_neighbour(points,centroids)
        centroids = find_centroids(points,centers)
        print("Centroids after iteration", i+1, ":", centroids.tolist())

    try:
        db_name = dict['DBNAME']
        saveToDB(centroids, db_name)
    except:
        print("Add parameter 'DBNAME' in order to save centroids to database.")
    return {"Centroids": centroids.tolist()}


