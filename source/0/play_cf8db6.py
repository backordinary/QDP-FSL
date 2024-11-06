# https://github.com/yajurahuja/QuantumCGT/blob/242ba3d037f813359baa9ca60bf8ec958ef56400/tictactoe/play.py
from graphics import *
from qiskit import *
from sys import *
from player import *
from gamep import *


if __name__ == "__main__":
	size  = 3
	winp = 3
	turn = 1
	#boxSize = min(100,int(750/size))
	a = game(size, winp, turn)
	a.setup()
	a.addplayer()
	a.addplayer()
	a.play()


