#!/usr/bin/env python
# coding: utf-8

"""Goban made with Python, pygame and go.py.

This is a front-end for my go library 'go.py', handling drawing and
pygame-related activities. Together they form a fully working goban.

"""

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

import pygame
import go
from sys import exit

BACKGROUND = 'images/ramin.jpg'
STONE_BLACK = 'images/black.png'
STONE_WHITE = 'images/white.png'
BOARD_SIZE = (820, 820)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Stone(go.Stone):
    def __init__(self, board, point, color):
        """Create, initialize and draw a stone."""
        super(Stone, self).__init__(board, point, color)
        self.coords = (self.point[0] * 40 - 15, self.point[1] * 40 - 15)
        self.draw()

    def draw(self):
        """Draw the stone as a circle."""
        if self.color == WHITE:
            screen.blit(white, self.coords)
        elif self.color == BLACK:
            screen.blit(black, self.coords)
        pygame.display.update()

    def remove(self):
        """Remove the stone from board."""
        blit_coords = (self.coords[0], self.coords[1])
        area_rect = pygame.Rect(blit_coords, (40, 40))
        screen.blit(background, blit_coords, area_rect)
        pygame.display.update()
        super(Stone, self).remove()


class Board(go.Board):
    def __init__(self):
        """Create, initialize and draw an empty board."""
        super(Board, self).__init__()
        self.outline = pygame.Rect(45, 45, 720, 720)
        self.draw()

    def draw(self):
        """Draw the board to the background and blit it to the screen.

        The board is drawn by first drawing the outline, then the 19x19
        grid and finally by adding hoshi to the board. All these
        operations are done with pygame's draw functions.

        This method should only be called once, when initializing the
        board.

        """
        pygame.draw.rect(background, BLACK, self.outline, 3)
        # Outline is inflated here for future use as a collidebox for the mouse
        self.outline.inflate_ip(20, 20)
        for i in range(18):
            for j in range(18):
                rect = pygame.Rect(45 + (40 * i), 45 + (40 * j), 40, 40)
                pygame.draw.rect(background, BLACK, rect, 1)
        for i in range(3):
            for j in range(3):
                coords = (165 + (240 * i), 165 + (240 * j))
                pygame.draw.circle(background, BLACK, coords, 5, 0)
        screen.blit(background, (0, 0))
        pygame.display.update()

    def update_liberties(self, added_stone=None):
        """Updates the liberties of the entire board, group by group.

        Usually a stone is added each turn. To allow killing by 'suicide',
        all the 'old' groups should be updated before the newly added one.

        """
        for group in self.groups:
            if added_stone:
                if group == added_stone.group:
                    continue
            group.update_liberties()
        if added_stone:
            added_stone.group.update_liberties()


def main():
    while True:
        pygame.time.wait(250)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and board.outline.collidepoint(event.pos):
                    x = int(round(((event.pos[0] - 5) / 40.0), 0))
                    y = int(round(((event.pos[1] - 5) / 40.0), 0))
                    stone = board.search(point=(x, y))
                    print(type(stone))
                    if stone is None or (type(stone).__name__ == 'list' and len(stone) == 0):
                        added_stone = Stone(board, (x, y), board.next)
                        liberties = added_stone.liberties()
                        if added_stone.group is not None:
                            group_liberties = added_stone.group.cal_liberties()
                        if len(liberties) > 0 or (group_liberties is not None and len(group_liberties) > 0):
                            board.turn()
                            board.update_liberties(added_stone)
                        else:
                            added_stone.remove()

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Goban')
    screen = pygame.display.set_mode(BOARD_SIZE, 0, 32)
    background = pygame.image.load(BACKGROUND).convert()
    black = pygame.transform.smoothscale(pygame.image.load(STONE_BLACK), (40, 40))
    white = pygame.transform.smoothscale(pygame.image.load(STONE_WHITE), (40, 40))
    board = Board()
    main()
