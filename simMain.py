#!/usr/bin/env python3

from simObjs import *
from simFuncs import get_colour
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np


# VOID clearly, but this is a good way to quickly modify the np.array
def add_domino_to_grid(domino, grid, order):
    ps = domino.points()
    cl = get_colour(domino, order)
    for p in ps:
        x, y = p.to_vec(order)
        grid[y - 1][x - 1] = cl


if __name__ == "__main__":

    # Get the order of the diamond to be created
    user_order = int(input("What is the order of the diamond (must be an integer): "))
    diamond = Diamond.bernoulli_diamond(user_order, 0.25)

    # Start building the grid to plot from the diamond
    bounds = [0, 2 * user_order, 2 * user_order, 0]
    grid = np.full((2 * user_order, 2 * user_order, 3), 255)
    for d in diamond.dominos:
        add_domino_to_grid(d, grid, user_order)
    plt.imshow(grid, extent=bounds)
    plt.axis("off")

    # Actually show the plot
    plt.show()

# Order 150 - 13 seconds
# Order 175 - 23 seconds
