# https://github.com/mirkoconsiglio/Dissertation/blob/0dd3faa8cdff47e67492f91c3f7a1eb6dc649071/data/ibmq_16_melbourne/2020.03.12_12.40.59_8192_22.5_(0.707%2B0j)_(0.707%2B0j)/state_city_plotter.py
import qiskit.visualization as visual
import pickle
import os
import numpy as np
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

plt.rcParams.update({'figure.max_open_warning': 0})
rc('font',**{'family':'Times New Roman','sans-serif':['Times New Roman'], 'size': 36})
rc('text', usetex=True)

def plot(theoretical_rho_GHZ_list, rho_GHZ_list, theoretical_rho_T_list, rho_T_list):
    try:
        os.mkdir('state_cities')
    except FileExistsError:
        pass

    c = ['#ba0c2f', '#0cba97']
    a = 0.8
    bbox = Bbox(np.array([[2.3, 0], [13.7, 4.2]]))

    for i in range(len(theoretical_rho_GHZ_list)):
        visual.plot_state_city(theoretical_rho_GHZ_list[i], color=c, alpha=a).savefig('state_cities/Theo_GHZ_state_city_{}.svg'.format(i), bbox_inches=bbox)
        visual.plot_state_city(rho_GHZ_list[i], color=c, alpha=a).savefig('state_cities/GHZ_state_city_{}.svg'.format(i), bbox_inches=bbox)
        visual.plot_state_city(theoretical_rho_T_list[i], color=c, alpha=a).savefig('state_cities/Theo_T_state_city_{}.svg'.format(i), bbox_inches=bbox)
        visual.plot_state_city(rho_T_list[i], color=c, alpha=a).savefig('state_cities/T_state_city_{}.svg'.format(i), bbox_inches=bbox)

data = pickle.load(open('data.json', 'rb'))

plot(data['theoretical_rho_GHZ_list'], data['rho_GHZ_list'], data['theoretical_rho_T_list'], data['rho_T_list'])