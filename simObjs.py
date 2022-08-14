#!/usr/bin/env python3

from itertools import chain
from functools import reduce
import simFuncs
import random as r
import math


class Point:
    """
    A simple implementation of a 2D coord point. This is used for building
    dominos and in proxy the diamond itself.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Return a tuple of the x,y coords (in the context of the lattice that the diamond
    # lives on)
    def to_vec(self, order):
        return (self.x + order, self.y + order)

    # Checks if two points are equal using there tuple representation
    def equals(self, other):
        return self.x == other.x and self.y == other.y

    # A positive value indicates self>other
    def compare(self, other):
        if self.x != other.x:
            return self.x - other.x
        else:
            return self.y - other.y

    # Return a new point which is the argument points added together
    def add_point(self, other):
        return Point(self.x + other.x, self.y + other.y)

    # Get a list of adjacent points (no diagonals)
    def adjacent_points(self):
        return list(
            map(
                lambda p: self.add_point(p),
                [Point(1, 0), Point(-1, 0), Point(0, 1), Point(0, -1)],
            )
        )


class Domino:
    """
    A domino is two adjacent Points put together. For convience the points that comprise
    the domino are sorted such that point1 < point 2
    """

    def __init__(self, point1, point2):
        # If point two is bigger it goes in the p2 field
        if point2.compare(point1) > 0:
            self.p1 = point1
            self.p2 = point2
        else:
            self.p1 = point2
            self.p2 = point1

    # A positive value indicates self>other
    def compare(self, other):
        if other == None:
            raise Exception("Cannot Compare None to a Domino")
        if not self.p1.equals(other.p1):
            return self.p1.compare(other.p1)
        else:
            return self.p2.compare(other.p2)

    # See if two dominos are the same
    def equals(self, other):
        if other == None:
            return False
        return self.p1.equals(other.p1) and self.p2.equals(other.p2)

    # Check if a domino is horizontal
    def is_hor(self):
        return (
            self.p1.y == self.p2.y
        )  # If the two points are the same height then horizontal

    # Check if a domino is vertical
    def is_ver(self):
        return (
            self.p1.x == self.p2.x
        )  # If the two points are the same across then vertical

    # Return a domino moved horizontally
    def hor_move(self, n):
        return Domino(self.p1.add_point(Point(n, 0)), self.p2.add_point(Point(n, 0)))

    # Return a domino moved vertically
    def ver_move(self, n):
        return Domino(self.p1.add_point(Point(0, n)), self.p2.add_point(Point(0, n)))

    # Identify the domino's orientation
    def domino_type(self, n):
        # The size of the diamond is added due to how the diamond is indexed in the code
        # versus on the lattice we normally think of it as occuping. (In the lattice, for our
        # shuffling to work we shift the domino to always be centered on (n,n) which also
        # makes sure the polar regions stay correct as when you shuffle the entire board needs
        # to shift - else north would be south and east would be west)
        match (self.is_hor(), (self.p1.x + self.p1.y + n) % 2 == 0):
            # Why are python match statements so weird?
            case (True, True):
                return "N"
            case (True, False):
                return "S"
            case (False, True):
                return "E"
            case (False, False):
                return "W"

    # Return a list of points covered by the dominos
    def points(self):
        return [self.p1, self.p2]

    # Return the minimum y value
    @staticmethod
    def y_min(x, n):
        if x > 0:
            return -n + x
        else:
            return -n + 1 - x

    # Return the minimum x value
    @staticmethod
    def x_min(y, n):
        if y > 0:
            return -n + y
        else:
            return -n + 1 - y

    # Change the vertical coords of a point, returning the coords (n is the order)
    @staticmethod
    def change_hor_coords(point, n):
        return (point.x - Domino.x_min(point.y, n), point.y + n - 1)

    # Change the vertical coords of a point, returning the coords (n is the order)
    @staticmethod
    def change_ver_coords(point, n):
        return (point.x + n - 1, point.y - Domino.y_min(point.x, n))


class Face:  # As described in the paper by Janvresse, Rue, and Velenik
    """
    Faces are the base of the checkboard on which the whole Aztec diamond lives. Each is
    indexed on the lattice by its bottom left point. Active faces are described by Janvresse, Rue, and Velenik
    are used in weight computation, and so their generation (for a given order) is included, but the
    weight computations are handled elsewhere.
    """

    def __init__(self, bottom_left_point):
        # The points that make the face are indexed as:
        # bottomleft, bottomright, topleft, topright
        self.points = [
            bottom_left_point,
            bottom_left_point.add_point(Point(1, 0)),
            bottom_left_point.add_point(Point(0, 1)),
            bottom_left_point.add_point(Point(1, 1)),
        ]
        # The possible dominos that can cover the face such that the face is the bottomleft
        # cell of a "block"
        self.horizontal_dominos = (
            Domino(self.points[0], self.points[1]),
            Domino(self.points[2], self.points[3]),
        )
        self.vertical_dominos = (
            Domino(self.points[0], self.points[2]),
            Domino(self.points[1], self.points[3]),
        )
        self.dominos = [
            self.horizontal_dominos[0],
            self.horizontal_dominos[1],
            self.vertical_dominos[0],
            self.vertical_dominos[1],
        ]

    # Each face has four weights associated with it, labelled starting at the topmost
    # edge and labelled clockwise with the first four greek letters (in the paper)
    def get_weights_of_faces(self, weights):
        return (
            weights.apply(self.horizontal_dominos[1]),  # alpha
            weights.apply(self.vertical_dominos[1]),  # beta
            weights.apply(self.horizontal_dominos[0]),  # gamma
            weights.apply(self.vertical_dominos[0]),  # delta
        )

    # Returns a random block in accordance to the probability distribution given by
    # the weights
    def random_block(self, weights):
        h0, h1 = self.horizontal_dominos
        v0, v1 = self.vertical_dominos

        # I'm not including greek unicode in the source, that is too much even
        # for me (vær så snill å tilgi meg for det)
        a, b, g, d = self.get_weights_of_faces(weights)

        up_down = a * g
        sides = b * d
        double_prod = up_down + sides

        # Watching for undefined division
        if double_prod == 0:
            raise ValueError("Double Product == 0 and therefore Not Tileable")
        # Simple cases where a prob == 0
        elif (a == 0.0) or (g == 0.0):
            return [v0, v1]
        elif (b == 0.0) or (d == 0.0):
            return [h0, h1]
        # If rand < p = (a * g)/(a * g + b * d)
        elif CustomGenWeight.bernoulli_return(up_down / double_prod):
            return [h0, h1]
        # And else!
        else:
            return [v0, v1]

    def next_diamond_construction(self, sub_diamond, weights):
        h0, h1 = self.horizontal_dominos
        v0, v1 = self.vertical_dominos

        # If only one matches, we fill in the other domino of the same kind (horizontal/
        # vertical). If two matches occur it must be either two horizontal or two vertical,
        # in which case we do nothing. All false means we fill a random square, and
        # other cases are impossible.
        match (
            sub_diamond.contains(h0),
            sub_diamond.contains(h1),
            sub_diamond.contains(v0),
            sub_diamond.contains(v1),
        ):
            # Case where the block is not at all tiled and we need to choose a random tiling
            # for it
            case (False, False, False, False):
                return self.random_block(weights)
            # Cases where the block is partially tiled
            case (True, False, False, False):
                return [h1]
            case (False, True, False, False):
                return [h0]
            case (False, False, True, False):
                return [v1]
            case (False, False, False, True):
                return [v0]
            # Cases where the block is already tiled
            case (True, True, False, False):
                return []
            case (False, False, True, True):
                return []
            # This should never be raised
            case _:
                raise Exception(
                    "Impossible Diamond Construction of order %d" % sub_diamond.order
                )

    # Returns the weights of the face for order n-1, and possible zeros to append
    # => ([(Domino, Double)], [Dominos])
    def sub_weights(self, weights):
        h0, h1 = self.horizontal_dominos
        v0, v1 = self.vertical_dominos

        a, b, g, d = self.get_weights_of_faces()
        double_prod = a * g + b * d

        # The general case for a non-zero double product
        if double_prod != 0:
            return_list = [
                (h0, a / double_prod),
                (h1, g / double_prod),
                (v0, b / double_prod),
                (v1, d / double_prod),
            ]
        else:
            # ALl cases where the double prod is zero
            match (a, b, g, d):
                case (0, 0, 0, 0):
                    const = 1 / math.sqrt(2)
                    return_list = [(h0, const), (h1, const), (v0, const), (v1, const)]
                case (_, _, 0, 0):
                    return_list = [
                        (h0, a),
                        (h1, 1 / (a + b)),
                        (v0, b),
                        (v1, 1 / (a + b)),
                    ]
                case (0, _, _, 0):
                    return_list = [
                        (h0, 1 / (b + g)),
                        (h1, g),
                        (v0, b),
                        (v1, 1 / (b + g)),
                    ]
                case (0, 0, _, _):
                    return_list = [
                        (h0, 1 / (d + g)),
                        (h1, g),
                        (v0, 1 / (d + g)),
                        (v1, d),
                    ]
                case (_, 0, 0, _):
                    return_list = [
                        (h0, a),
                        (h1, 1 / (a + d)),
                        (v0, 1 / (a + d)),
                        (v1, d),
                    ]
                case _:
                    raise Exception("Case Break Calculating Sub Weights")

        # Yes it's a mess, but I don't know how else to sort of "walk around the square"
        # checking all the conditions. Again, this feels like me not having python-fu.
        return_zeros_lists = []
        if a == 0 and b == 0:
            return_zeros_lists.append([h1.hor_move(1), v1.ver_move(1)])
        else:
            return_zeros_lists.append([])
        if b == 0 and g == 0:
            return_zeros_lists.append([h0.hor_move(1), v1.ver_move(-1)])
        else:
            return_zeros_lists.append([])
        if g == 0 and d == 0:
            return_zeros_lists.append([h0.hor_move(-1), v0.ver_move(-1)])
        else:
            return_zeros_lists.append([])
        if d == 0 and a == 0:
            return_zeros_lists.append([h1.hor_move(-1), v0.ver_move(1)])
        else:
            return_zeros_lists.append([])

        return (return_list, list(chain(*return_zeros_lists)))

    @staticmethod
    def active_faces(n):
        faces = []
        for j in range(0, -n, -1):
            for k in range(n):
                faces.append(Face(Point(j + k, -n + 1 - j + k)))
        return faces


class Diamond:
    """
    A representation of the Aztec Diamond and a tiling. The Diamond A_n should have a
    n(n+1) dominos. Lists represent columns of the Diamond, and will follow the pattern:
    2,4,6,...,2n-2,2n,2n,2n-2,...,6,4,2 (in terms of there sizes)

    Ideally the values in the vector would be like a `Maybe Domino` in Haskell or
    `Result<Domino>` in Rust, be for now we are just going to use the keyword None
    to reprecent that a cell in a vector is covered by another domino (as each domino
    covers two cells of the Diamond).
    """

    def __init__(self, dominos):
        # The dominos in the argument is a list of lists holding the dominos and None values

        # The diamond list is the raw list of lists that comprise the domino
        self.diamond_lists = dominos
        # The number of dominos
        self.dominos_num = sum(
            list(
                map(
                    lambda ds: len(list(filter(None, ds))),
                    self.diamond_lists,
                )
            )
        )
        # A single flattened list of dominos
        self.dominos = simFuncs.flatMap(
            lambda l: list(filter(None, l)), self.diamond_lists
        )
        # The order of the diamond gotten by the quadratic formula
        self.order = int((-1 + (1 + 4 * self.dominos_num) ** (1 / 2)) / 2)

    # Search the diamond to see if a specific dominos is contained
    def contains(self, domino):
        if not self.in_bounds_domino(domino):
            return False
        x, y = Domino.change_ver_coords(domino.p1, self.order)
        if domino.equals(self.diamond_lists[x][y]):
            return True
        else:
            return False

    # The only reason these are not static is because of how I want to implement these
    # later (possibly in a different language)

    # Check if a point is in bounds (a use of the taxicab metric, huzzah!)
    def in_bounds_point(self, point):
        return abs(point.x - 0.5) + abs(point.y - 0.5) <= self.order

    # Check if a domino is in bounds using the above
    def in_bounds_domino(self, domino):
        if domino == None:
            return False
        return self.in_bounds_point(domino.p1) and self.in_bounds_point(domino.p2)

    # Generate a random tiling of order weights[-1].n
    #
    # NOTE: Okay, programmer talk incoming. Python isn't lazy, and this is some nasty
    # recursion that is going to gum things up a bit. Particularly, under the hood of
    # reduce what is really going on is a left fold, but it beats writing out the for
    # loops by hand. Also, if one part isn't going to parallize well, it will be this.
    #
    # NOTE: What's really happening in the fold is that a list of Weights is iteratively
    # folding into a list of diamonds.
    @staticmethod
    def generate_diamond(weights):
        return reduce(
            lambda diamond, weight: weight.generate_diamond(diamond),
            weights[1:],
            weights[0].generate_order_one(),
        )

    # Uniform diamond generation just takes an order
    def uniform_diamond(n):
        return Diamond.generate_diamond(
            list(map(lambda x: UniformWeightGeneration(x), list(range(1, n + 1))))
        )


"""
This is for more generic distributions using the weight algorithm described in the paper
by Janvresse, Rue, and Velenik. For the moment this is incomplete (as getting a working
prototype was more important) but after flushing out the face all what is needed is
to complete the weight generation.

Not only are more probability distributions going to be added, most if not all of this
code needs a major refractor. The current plan is to test a few more probability distributions,
and then do a major rewrite in Haskell introducing a very important, non-math feature:
parallelization. However, this is an imperfect world, with imperfect code, and so for now
this is how it will stay, for now.
"""


class Weight:
    """
    Think of this like a Haskell class (or I guess like a java inteface... kinda)
    An integer, natuarally. But this is the best python can do for an "abstract class".
    """

    # I am almost certain that this is bad practice. Just roll with it...
    def __init__(self, n):
        self.n = n

    # Returns the weight
    def apply(self, domino):
        pass

    # Updates the weight (VOID)
    def update(self, domino, weight):
        pass

    # Return the weight for the subdiamond.
    def sub_weights(self):
        pass

    # Check if a point is in bounds (a use of the taxicab metric, huzzah!)
    def in_bounds_point(self, point):
        return abs(point.x - 0.5) + abs(point.y - 0.5) <= self.n

    # Check if a domino is in bounds using the above
    def in_bounds_domino(self, domino):
        if domino == None:
            return False
        return in_bounds_point(domino.p1) and in_bounds_point(domino.p2)


class CustomWeight(Weight):
    """
    This implementation is not optimised and things are stored as a matter of convinence to me
    rather than ease of visualisation. So long as the apply and update methods are consistant
    with each other it should be fine, but a rewrite in the future would probably be smart.

    NOTE: The sub_weights method of the base weight is not overriden as this is not meant to
    be used outside of internal class methods (like it is used in the Face class). I must
    admit, that python let's you only "half" de-abstract a class is pretty neat.
    """

    def __init__(n):
        super().__init__(n)

        # Okay, we're reusing a method that really isn't built for this, but it works.
        self.horizontal_weights = simFuncs.empty_domino_array(self.n)
        self.vertical_weights = simFuncs.empty_domino_array(self.n)

    def apply(self, domino):
        if domino.is_hor:
            x, y = Domino.change_hor_coords(domino.p1, self.n)
            return self.horizontal_weights[y][x]
        else:
            x, y = Domino.change_ver_coords(domino.p1, self.n)
            return self.vertical_weights[x][y]

    def update(self, domino, weight):
        if domino.is_hor:
            x, y = Domino.change_hor_coords(domino.p1, self.n)
            self.horizontal_weights[y][x] = weight
        else:
            x, y = Domino.change_ver_coords(domino.p1, self.n)
            self.vertical_weights[x][y] = weight


class GenWeight(Weight):
    """
    Generates an aztec diamond given the weights and a diamond of order n-1. Again, this
    is basically just an abstract class.
    """

    # This should apply the algorithm and return a new Diamond of order n given a sub_diamond
    # of order n-1

    def __init__(self, n):
        super().__init__(n)

    def generate_diamond(self, sub_diamond):
        dominos = simFuncs.empty_domino_array(self.n)

        for face in Face.active_faces(self.n):
            new_doms = face.next_diamond_construction(sub_diamond, self)
            for dom in new_doms:
                x, y = Domino.change_ver_coords(dom.p1, self.n)
                dominos[x][y] = dom

        return Diamond(dominos)

    # Well we need to start somewhere
    def generate_order_one(self):
        if self.n == 1:
            dominos = Face(Point(0, 0)).random_block(self)
            dom1, dom2 = dominos[0], dominos[1]
            if dom1.is_hor():
                return Diamond([[dom1, dom2], [None, None]])
            else:
                return Diamond([[dom1, None], [dom2, None]])
        else:
            raise ValueError(
                "The Weight obj is wrong. It should be for order 1 not %d" % n
            )


class CustomGenWeight(GenWeight, CustomWeight):
    """
    Again, as with Custom Weight, this is an internal class. Truthfully everything should
    be fairly self explanitory.

    NOTE: Fully implements all methods and fields of Weight as is a CONCRETE CLASS
    """

    # This is probably bad practice, but I really do want n (the order) to be an instance
    # variable and as python doesn't have an abstract keyword this is what we are working with
    def __init__(self, order):
        super().__init__(order)

    def sub_weights(self):
        if n == 1:
            raise ValueError("Weight obj cannot be made for an order 0 diamond")
        else:
            new_weights = CustomGenWeight(n - 1)

            for domino in filter(
                lambda d: new_weights.in_bounds_domino(d),
                simFuncs.flatMap(
                    lambda f: CustomGenWeight.sub_weight_helper(f, new_weights)
                ),
            ):
                new_weights.update(d, 0.0)

            return new_weights

    # Oh Haskell how I miss your where keyword so as to not pollute my namespace.
    # At least the helper is bound to the scope of the class.
    @staticmethod
    def sub_weight_helper(face, weights):
        new_pairs, new_zeros = face.sub_weights(self)

        # NON-PURE!
        for (dom, wei) in new_pairs:
            if weights.in_bounds_domino(d):
                weights.update(dom, wei)

        return new_zeros

    # Given a probability p, we generate a random double and return true if it is < p
    @staticmethod
    def bernoulli_return(p):
        return r.random() < p


class UniformWeightGeneration(GenWeight):
    """
    Currently uniform weight is the only supported "generation" for an Aztec Diamond. I'll
    add some more later, but for now this is going to be what is included.
    """

    # Standard
    def __init__(self, order):
        super().__init__(order)

    # An important part of Uniform Generation is that the weights should never change
    # (they are uniform) and so the update method throws a VERY noise error
    def update(self, domino, weight):
        raise NotImplementedError("Uniform Weights Do Not Change")

    # Again, Uniform weights
    def apply(self, domino):
        return 1.0

    # The subweights are super easy to compute, again because everything is Uniform
    def sub_weights(self):
        if n == 1:
            raise ValueError("Weight obj cannot be made for an order 0 diamond")
        else:
            return UniformWeightGeneration(n - 1)


"""
I'll need this later, but for now we aren't doing inverse operations and thus it is
not required. For now, just ignore this code.
"""

# class DiamondConstruction:
#     """
#     Here my true detest for OOP will shine through. Diamond construction is done through
#     this seperate class as the Diamond will be used only for the algorithm.
#     """

#     def __init__(self, order):
#         self.order = order
#         self.dominos = simFuncs.empty_domino_array(order)

#     # Scale a domino to the order of the Diamond being constructed as apply the ver_change
#     # and then add it to the construction
#     def update(self, domino):
#         coords = Domino.change_ver_coords(domino.p1, order)
#         self[coords[0], coords[1]] = domino

#     # Count the number of dominos in the construction
#     def dominos_number(self):
#         len(list(filter(None, list(chain(self.dominos)))))

#     # Check if a point is in bounds (a use of the taxicab metric, huzzah!)
#     def in_bounds_point(self, point):
#         return abs(point.x - 0.5) + abs(point.y - 0.5) <= self.order

#     # Check if a domino is in bounds using the above
#     def in_bounds_domino(self, domino):
#         if domino == None:
#             return False
#         return in_bounds_point(domino.p1) and in_bounds_point(domino.p2)

#     # See if a domino is contained in the construction at the position of the argument
#     def contains(self, domino):
#         if in_bounds_domino(domino) == False:
#             return False
#         coords = Domino.change_ver_coords(domino.p1, order)
#         if self.dominos[coords[x]][coords[y]] == None:
#             return False
#         else:
#             return True

#     # See if a given point in the construction is occupied
#     def is_point_occupied(self, point):
#         # A point is covered if there is a domino that covers it AND an adjacent point
#         possible_covering_dominos = [
#             Domino(point, point + Point(1, 0)),
#             Domino(point, point + Point(-1, 0)),
#             Domino(point + Point(-1, 0), point),
#             Domino(point + Point(1, 0), point),
#         ]
#         for d in possible_covering_dominos:
#             if self.contains(d):
#                 return True
#             else:
#                 return False

#     # Returns a list of possible dominos cover a given point (or None should the point
#     # already be occupied, as then no dominos "could" cover the point as one already does).
#     def possible_dominos_on(self, point):
#         if is_point_occupied(point):
#             return None
#         return list(
#             map(
#                 lambda p: Domino(p, point) if point.compare(p) else Domino(point, p),
#                 filter(
#                     lambda p: self.is_point_occupied(p),
#                     filter(lambda p: self.in_bounds_point(p), point.adjacent_points()),
#                 ),
#             )
#         )

#     # Get a list of the dominos forced by the possible dominos on the construction
#     # This is, as suggested, EXACTLY the dominos that are forced to exist. If that is
#     # nebulous, see the inner filter expression.
#     def forced_dominos(self):
#         return simFuncs.distinct_domino_list(
#             list(
#                 chain(
#                     filter(
#                         lambda doms: (len(ds) != 0) and (len(ds[1:]) == 0),
#                         map(
#                             lambda p: self.possible_dominos_on(p),
#                             DiamondConstruction.all_points(),
#                         ),
#                     )
#                 )
#             )
#         )

#     # Fill the construction with all the dominos that are forced by the current state
#     # Just a utility VOID function
#     def fill_forced(self):
#         doms_to_fill = self.forced_dominos()
#         while len(doms_to_fill) != 0:
#             for d in doms_to_fill:
#                 self.update(d)
#             # We use the while loop as introducing forced dominos may make other possibilities
#             # become forced dominos (think like a soduko game)
#             doms_to_fill = self.forced_dominos()

#     # Insert a diamond into a construction (VOID obviously)
#     def insert_diamond(self, diamond, center=Point(0, 0)):
#         doms_for_construction = list(
#             map(
#                 lambda d: Domino(d.p1 + center, d.p2 + center),
#                 diamond.list_of_dominos(),
#             )
#         )
#         for d in doms_for_construction:
#             self.update(d)

#     # Turn a construction into a diamond
#     def to_diamond(self):
#         return Diamond(self.dominos)

#     # Returns a list of all the points of an order n Diamond
#     @staticmethod
#     def all_points(order):
#         # Should probably do a list comprehension or something
#         points = []
#         for y in range(1, self.order + 1):
#             for x in range(-self.order + y, (self.order + 1) - y + 1):
#                 points.append(Point(x, y))
#         return list(chain(map(lambda p: [Point(p.x, p.y), Point(x, -y + 1)])))
