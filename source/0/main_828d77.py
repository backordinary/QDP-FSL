# https://github.com/CCNYseniors/rubik/blob/92ea9cbe816d85db447b8c6cdedf42b4e3f4e7c1/main.py
import pygame
import random
import time
import numpy as np
from qiskit import QuantumCircuit, execute, BasicAer, ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram

from constants import CENTERS, EDGES, CORNERS
from constants import F, B, R, L, U, D, CW, ACW
from constants import COLORS, OPPOSITE, COLORS_MAP
from constants import MOVE, MOVE2LAYERS, ROTATE

BKGND = (1, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHTBLUE = (0, 176, 240)

RED = (255, 0, 0)
ORANGE = (255, 127, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
GREAN = (0, 96, 0)
BLUE = (0, 0, 255)
INDIGO = (46, 43, 95)
VIOLET = (139, 0, 255)
colorlst = [VIOLET, INDIGO, BLUE, GREEN, YELLOW, ORANGE, RED]

Move = {
    CW: {
        F: {U: R, R: D, D: L, L: U},
        U: {B: R, R: F, F: L, L: B},
        R: {U: B, B: D, D: F, F: U},
        B: {U: L, L: D, D: R, R: U},
        L: {U: F, F: D, D: B, B: U},
        D: {L: F, F: R, R: B, B: L},
    },
    ACW: {
        F: {U: L, L: D, D: R, R: U},
        U: {B: L, L: F, F: R, R: B},
        R: {F: D, D: B, B: U, U: F},
        B: {L: U, U: R, R: D, D: L},
        L: {D: F, F: U, U: B, B: D},
        D: {B: R, R: F, F: L, L: B},
    }
}

Rotate = {
    CW: {
        R: {F: U, U: B, B: D, D: F},
        F: {U: R, R: D, D: L, L: U},
        U: {F: L, L: B, B: R, R: F},
        L: {F: D, D: B, B: U, U: F},
        B: {U: L, L: D, D: R, R: U},
        D: {F: R, R: B, B: L, L: F},
    },
    ACW: {
        R: {F: D, D: B, B: U, U: F},
        F: {U: L, L: D, D: R, R: U},
        U: {F: R, R: B, B: L, L: F},
        L: {F: U, U: B, B: D, D: F},
        B: {U: R, R: D, D: L, L: U},
        D: {F: L, L: B, B: R, R: F},
    }
}


class Cubelet:
    def __init__(self):
        pass

    def move(self, direction, face):
        pass

    def rotate(self, direction, face):
        pass


class Center(Cubelet):
    def __init__(self, position, color):
        self.position = position
        self.color = color

    def rotate(self, direction, face):
        if self.position in Rotate[direction][face]:
            self.position = Rotate[direction][face][self.position]

    def if_position(self, position):
        if self.position == position:
            return self.color

    def __repr__(self):
        return f'Center({self.position}, {COLORS_MAP[self.color]})'


class MCubelet(Cubelet):
    ORDER = [R, L, U, D, F, B]

    def __init__(self, positions, colors):
        self.positions = positions
        self.colors = colors

    def reorder(self):
        # This is necessary after any transformation as the enforced tuple ordering may be destroyed.
        # and this method enforces it.
        temp = sorted(zip(self.positions, self.colors),
                      key=lambda pc: MCubelet.ORDER.index(pc[0]))
        self.positions = tuple(p for p, _ in temp)
        self.colors = tuple(c for _, c in temp)

    def move(self, direction, face):
        # print(direction, face, self.positions)
        if face in self.positions:
            self.positions = tuple(
                f if face == f else Move[direction][face][f] for f in self.positions)
            self.reorder()

    def rotate(self, direction, face):
        self.positions = tuple(Rotate[direction][face].get(
            f, f) for f in self.positions)
        self.reorder()

    def if_position(self, positions):
        if self.positions == positions:
            return self.colors


class Edge(MCubelet):
    def __repr__(self):
        return f'Edge({self.positions}, {tuple(COLORS_MAP[c] for c in self.colors)})'


class Corner(MCubelet):
    def __repr__(self):
        return f'Corner({self.positions}, {tuple(COLORS_MAP[c] for c in self.colors)})'


def log(func):
    def inner(*args, **kwargs):
        x = func(*args, **kwargs)
        print('INPUTS:', args, kwargs, 'OUTPUTS:', x)
        return x

    return inner


class Rubik:
    def __init__(self):
        self.centers = [Center(p, COLORS[p]) for p in CENTERS]
        self.edges = [Edge((p1, p2), (COLORS[p1], COLORS[p2]))
                      for p1, p2 in EDGES]
        self.corners = [Corner(
            (p1, p2, p3), (COLORS[p1], COLORS[p2], COLORS[p3])) for p1, p2, p3 in CORNERS]
        self.pieces = [*self.centers, *self.edges, *self.corners]

    def move(self, direction, face, times=1):
        for _ in range(times):
            for piece in self.pieces:
                piece.move(direction, face)

    def rotate(self, direction, face, times=1):
        for _ in range(times):
            for piece in self.pieces:
                piece.rotate(direction, face)

    def move2layers(self, direction, face, times=1):
        self.rotate(direction, face, times)
        self.move(direction, OPPOSITE[face], times)

    def transform(self, direction, face, action, times=1):
        if action == ROTATE:
            self.rotate(direction, face, times)
        elif action == MOVE:
            self.move(direction, face, times)
        elif action == MOVE2LAYERS:
            self.move2layers(direction, face, times)

    def get_colors(self, positions):
        if not isinstance(positions, tuple):
            for center in self.centers:
                if center.if_position(positions):
                    return center.if_position(positions)
        elif len(positions) == 2:
            for edge in self.edges:
                if edge.if_position(positions):
                    return edge.if_position(positions)
        elif len(positions) == 3:
            for corner in self.corners:
                if corner.if_position(positions):
                    return corner.if_position(positions)

    """
        Following methods returns the requested cubelet identified either by color or position.
        Pass only one of color(s) or position(s). Also note is method positions can be in any order unlike other methods.
    """

    def get_center(self, color=None, position=None):
        for center in self.centers:
            if center.color == color or center.position == position:
                return center

    def get_edge(self, colors=None, positions=None):
        for edge in self.edges:
            if (colors is not None and set(edge.colors) == set(colors)) or (
                    positions is not None and set(edge.positions) == set(positions)):
                return edge

    def get_corner(self, colors=None, positions=None):
        for corner in self.corners:
            if (colors is not None and set(corner.colors) == set(colors)) or (
                    positions is not None and set(corner.positions) == set(positions)):
                return corner


######################################################################
pygame.init()
clock = pygame.time.Clock()

qubitN = 3

bricksgap = 5
brickswid = 40
bricksdep = 40
brickspit = brickswid + bricksgap
brickMax = 2**qubitN
brickGroupXShift = 20

brickGroupYShift = bricksdep + bricksgap + brickGroupXShift


dockYdep = 80
dockYpos = brickGroupYShift + 5*(bricksdep + bricksgap) + bricksgap*2


scorXpos = brickGroupXShift*2+brickMax*brickspit-bricksgap
scorXwid = 200
circYpos = dockYdep + dockYpos
circYdep = 300

# Open a new window
sizeX = scorXpos + scorXwid
sizeY = circYpos + circYdep + brickGroupXShift
size = (sizeX, sizeY)  # Width, height
win = pygame.display.set_mode((size))
pygame.display.set_caption("Rubik's Cube")

# rubik = Rubik()
points, centers, edges, corners = get_init_points()
save_positions, reset_positions = handle_save_points(points)
handle_mouse_drag = init_mouse_drag(points)
in_progress_animation, init_move, animate = animation(
    rubik, centers, edges, corners)
handle_functional_keys, in_progress_function, continue_function = init_functional_keys(
    rubik, init_move)
handle_key_event = init_handle_keys(init_move, save_positions, reset_positions)

run = True
while run:
    clock.tick(60)
    for event in pygame.event.get():
            if event.type == pygame.QUIT or \
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or \
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                run = False
            if not in_progress_animation() and not in_progress_function():
                handle_mouse_drag(event)
                handle_key_event(event)
                handle_functional_keys(event)
            if in_progress_animation():
                animate()
            elif in_progress_function():
                continue_function()
            else:
                handle_rotation_keys(points)

     # Draw a line       
    pygame.draw.line(screen, WHITE, [0, brickGroupXShift], [
                     scorXpos, brickGroupXShift], 2)
    pygame.draw.line(screen, WHITE, [0, dockYpos], [scorXpos, dockYpos], 2)
    pygame.draw.line(screen, WHITE, [0, circYpos], [scorXpos, circYpos], 2)
    pygame.draw.line(screen, WHITE, [scorXpos, 0], [scorXpos, sizeY], 2)
    pygame.draw.line(screen, WHITE, [qbt0PosX+brickspit//2-1, qbt0PosY], [
                     qbt0PosX+brickspit//2-1, qbt0PosY+circYdep//8*6], 2)
    pygame.draw.line(screen, WHITE, [qbt0PosX-brickspit//2-1, qbt0PosY], [
                     qbt0PosX-brickspit//2-1, qbt0PosY+circYdep//8*6], 2)
    pygame.draw.line(screen, WHITE, [qbt1PosX+brickspit//2-1, qbt1PosY], [
                     qbt1PosX+brickspit//2-1, qbt1PosY+circYdep//8*6], 2)
    pygame.draw.line(screen, WHITE, [qbt1PosX-brickspit//2-1, qbt1PosY], [
                     qbt1PosX-brickspit//2-1, qbt1PosY+circYdep//8*6], 2)
    pygame.draw.line(screen, WHITE, [qbt2PosX+brickspit//2-1, qbt2PosY], [
                     qbt2PosX+brickspit//2-1, qbt2PosY+circYdep//8*6], 2)
    pygame.draw.line(screen, WHITE, [qbt2PosX-brickspit//2-1, qbt2PosY], [
                     qbt2PosX-brickspit//2-1, qbt2PosY+circYdep//8*6], 2)

    font = pygame.font.SysFont('arial', 12) 
