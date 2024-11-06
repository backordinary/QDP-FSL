# https://github.com/MichaelCullen2011/QuantumAndNeutrinos/blob/074d0a5178043d82f42d6833b656b8a6ecc78c09/.history/blochsphere_20210209155410.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import plot_bloch_vector, plot_bloch_multivector, plot_state_city, plot_state_paulivec, plot_state_qsphere

# from pylab import *
from qutip import *

from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt

import imageio
from PIL import Image
import pyglet
import os

'''
Using Qiskit
'''


def qiskit():
    # Plotting Single Bloch Sphere
    plot_bloch_vector([0, 1, 0], title='Bloch Sphere')

    # Building Quantum Circuit to use for multiqubit systems
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Plotting Multi Bloch System
    state = Statevector.from_instruction(qc)
    plot_bloch_multivector(state, title="New Bloch Multivector")

    # Plotting Bloch City Scape
    plot_state_city(state, color=['midnightblue', 'midnightblue'],
                    title="New State City")

    # Plotting Bloch Pauli Vectors
    plot_state_paulivec(state, color='midnightblue',
                        title="New PauliVec plot")



'''
Using qutip
'''


class Qutip:
    def __init__(self, states, duration=0.1, save_all=False):
        # Defining a Bloch
        b = Bloch()
        b.vector_color = ['r']
        b.view = [-40, 30]
        images = []
        try:
            length = len(states)
        except:
            length = 1
            states = [states]

        # Normalise Colours
        norm = mpl.colors.Normalize(0, length)
        colors = cm.cool(norm(range(length)))    # cool, summer, winter, autumn

        # Sphere Properties
        b.point_color = list(colors)
        b.point_marker = ['x']
        b.point_size = [30]

        # Defining save destination
        dir = './blochsphere_images/'
        for i in range(length):
            b.clear()
            b.add_states(states[i])
            b.add_states(states[:(i + 1)], 'point')
            if save_all:
                b.save(dirc=dir)
                filename = "temp/bloch_{i}01d.png".format(i=i)
            else:
                filename = "temp_file.png"
                b.save(dir + filename)
            images.append(imageio.imread(dir + filename))
        imageio.mimsave(dir + 'gif/' + 'bloch_anim.gif', images, duration=duration)
        im = Image.open(dir + 'gif/' + 'bloch_anim.gif')

        animation = pyglet.image.load_animation(dir + 'gif/' + 'bloch_anim.gif')
        sprite = pyglet.sprite.Sprite(animation)

        w = sprite.width
        h = sprite.height
        window = pyglet.window.Window(width=w, height=h)
        r, g, b, alpha = 0.5, 0.5, 0.8, 0.5
        pyglet.gl.glClearColor(r, g, b, alpha)

        @window.event
        def on_draw():
            window.clear()
            sprite.draw()

        pyglet.app.run()


states = []
thetas = linspace(0, pi, 21)
for theta in thetas:
    states.append((cos(theta/2) * basis(2, 0) + sin(theta/2) * basis(2, 1)).unit())


# qiskit()
Qutip(states, duration=0.1, save_all=False)

plt.show()











