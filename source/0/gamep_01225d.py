# https://github.com/yajurahuja/QuantumCGT/blob/242ba3d037f813359baa9ca60bf8ec958ef56400/tictactoe/gamep.py
from graphics import *
from qiskit import *
from sys import*
from player import*

class game:
	def __init__(self, size, winp, turn):
		self.boardsize = size
		self.win_parameter = winp
		self.turn = 1
		self.game_circuit = QuantumCircuit(self.boardsize ** 2, self.boardsize ** 2) # nXn size quantum circuit
		self.simulator = Aer.get_backend('qasm_simulator')
		self.board = [[ 0 for i in range(size)] for j in range(size)] # board matrix
		self.turn = 1
		self.moves = 0	
		self.points = []
		self.control_positions = {}
		self.target_positions = {}
		self.choice_info = ''
		self.players = []

	def addplayer(self):
		self.players.append(qplayer(len(self.players) + 1, self))

	def setup(self):
		self.initsetup()
		for i in range(self.boardsize ** 2):
			self.game_circuit.h(i)

	def initsetup(self):
		self.choice_info = "Choose an move (1-3): \n 1. Classical Move \n 2. Quantum Move \n 3. Reset Move\n"

	def play(self):
		while True:
			choice = self.playerchoice()
			self.move(choice)
			status = self.check_winner()
			if status == 1:
				print ("Player 1 wins")
				break
			elif status == 2:
				print ("Player 2 wins")
				break
			elif len(self.points) == self.boardsize ** 2:
				print ("Draw")
				break

	def playerchoice(self):
		print(" Turn: Player " + str(self.turn))
		return int(input(self.choice_info))

	def move(self, choice):
		self.players[self.turn - 1].move(choice)
		self.print_gameboard()
		self.turn = 3 - self.turn
		self.moves = self.moves + 1

	def update(self, index):
		result = execute(self.game_circuit, backend = self.simulator, shots=1).result()
		circuit_array = list(result.get_counts(self.game_circuit).keys())[0]
		print("Circuit Array: " + str(circuit_array))
		for i in self.points:
			row = int((i - 1)/self.boardsize)
			column = int((i - 1)%self.boardsize)
			if(circuit_array[i - 1] == '0' and self.board[row][column] != 1):
				self.board[row][column] = 1
			elif(circuit_array[i - 1] == '1' and self.board[row][column] != 2):
				self.board[row][column] = 2
		if index in self.control_positions.keys():
			for target in self.control_positions[index]:
				self.target_positions[target].remove(index)
				if self.target_positions[target] == []:
					del self.target_positions[target]
			del self.control_positions[index]

		print(self.control_positions)
		print(self.target_positions)

	def boolp(self, x, y, n):
		if x >= self.boardsize or y >= self.boardsize or x<0 or y<0: 
			return False
		return n == self.board[x][y]

	def check_winner(self):
		for row in range(self.boardsize):
			for column in range(self.boardsize):
				box = self.board[row][column]
				if box == 0: continue

				#Check row
				check = True
				for i in range(self.win_parameter):
					check = self.boolp(row , column+i, box)
					if not check: break
				if check: return box

				#Check column
				check = True
				for i in range(self.win_parameter):
					check = self.boolp(row + i, column, box)
					if not check: break
				if check: return box 
				
				#Check diagonal1
				check = True
				for i in range(self.win_parameter):
					check = self.boolp(row + i, column + i, box)
					if not check: break
				if check: return box

				#Check diagonal2
				check = True
				for i in range(self.win_parameter):
					check = self.boolp(row - i , column + i, box)
					if not check: break
				if check: return box

		return 0

	def print_gameboard(self):
		for row in self.board:
			print(row)