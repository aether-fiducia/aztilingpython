#!/usr/bin/env python3
import simObjs
from itertools import chain

# This isn't best practice, but it saves from having a bunch
# of nasty inheritence in the objects

# Currently the colours are just RGB values but this may change
def get_colour(domino, order):
    match domino.domino_type(order):
        case "N":
            return [0, 0, 255]
        case "S":
            return [255, 0, 0]
        case "E":
            return [0, 255, 0]
        case "W":
            return [255, 255, 0]


# Make an empty list of lists of Nones to serve as a base for dominos. Proper fixed
# sized arrays would be faster here, but my python-fu is a little rough so I'm not sure
# how'd I would implement that.
def empty_domino_array(order):
    dominos = [[] for j in range(2 * order)]
    for j in range(order):
        dominos[j] = [None for j in range(2 * (j + 1))]
        dominos[2 * order - 1 - j] = [None for j in range(2 * (j + 1))]
    return dominos


# Filter a list of dominos down to distinct elements
def distinct_domino_list(dominos):
    distinct = []
    for d in dominos:
        if len(list(filter(lambda x: d.equals(x), distinct))) == 0:
            distinct.append(d)
    return distinct


# Oh what I would give for python to have proper higher order function support. For now,
# this shouldn't be a bottleneck
def flatMap(func, list_of_lists):
    return list(chain(*list(map(lambda l: func(l), list_of_lists))))


def active_faces(n):
    faces = []
    for j in range(0, -n, -1):
        for k in range(n):
            faces.append(simObjs.Face(simObjs.Point(j + k, -n + 1 - j + k)))
    return faces
