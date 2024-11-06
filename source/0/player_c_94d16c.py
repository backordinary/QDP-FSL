# https://github.com/yajurahuja/QuantumCGT/blob/317972c7c27a545f4928a76134b2b35531b8f57c/Qtictactoe/player_c.py
from graphics import *
from qiskit import *
from sys import*
import numpy as np



class player:
	
	def __init__(self, number, game):
		self.player_number = number
		self.game = game

	def move(self):
		print("Please input as box number(1-9) probability(0-1) and press enter. type d and press enter to end turn.")
		box_number = []
		prob = []
		while True:
			inp = input()
			if (inp == 'd'):
				break
			inp = inp.split(' ')
			print(inp)
			box_number.append(int(inp[0]))
			prob.append(float(inp[1]))
		print(box_number, prob)
		if(sum(prob) != 1):
			print("Error: probability sum not equal to 1")
			return self.move()
		else:
			for i in range(len(box_number)):
				row, column = (box_number[i] - 1)//3, (box_number[i] - 1)%3
				if(prob[i] > self.game.board[row][column][0]):
					print("Error: probability entered is more than empty")
					return self.move()
		return box_number, prob

		