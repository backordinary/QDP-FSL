# https://github.com/theheavygluon/geoBloch/blob/2b0453308f2a2da3e80458083d821d68be343ea6/geoBloch.py
import numpy as np
import pandas as pd
import geopy
from geopy.geocoders import Nominatim
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_vector

def getLocation(cityname):
    geolocator = Nominatim(user_agent='my-app')
    location = geolocator.geocode(cityname)
    lat = location.latitude
    lng = location.longitude
    return lat,lng

def conv(phi):
    if (phi < 0): return np.pi - abs(phi)
    else: return np.pi/2 - phi
    
def geoBloch(address):
    coords = getLocation(address)
    theta, phi, = coords[0]*np.pi/180, coords[1]*np.pi/180
    
    return plot_bloch_vector([1,conv(theta),phi], title=address+" on Bloch sphere", coord_type='spherical')

def statevecToAddress(state):
    return "Feature under construction"
    
