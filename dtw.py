import pandas as pd
import numpy as np


def ecl(r, o):
    """
    :param r: point reference A 2d point with x and y
    :param o: point observed A 2d point with x and y
    :return: Euclidean distance between te two points
    """
    return ((r[0] - o[0]) ** 2 + (r[1] - o[1]) ** 2) ** 0.5

def dtw(r, o, w=2):
    """
    :param r: num of the reference signal's file => a 2 dimensions  x y vector
    :param o: nom of the observed signal's file => a 2 dimensions x y  vector
    :param w: weight for diagonal transition, in our project it's fixed to 2
    :return: scalar = to the lowest cumulative difference between the two signals
    """

    ## Creation of the cost matrix
    cost_m = np.zeros((len(r), len(o)))  # Create the cost matrix

    for ir in range(len(r)):
        for io in range(len(o)):
            # First: Origin
            if io == ir == 0:
                cost_m[0, 0] = ecl(r[ir], o[io])
            # Then let's continue with the first row or first column
            elif ir == 0:
                cost_m[ir, io] = cost_m[ir, io - 1] + ecl(r[ir], o[io])
            elif io == 0:
                cost_m[ir, io] = cost_m[ir - 1, io] + ecl(r[ir], o[io])
            # Finish with the rest of the matrix
            else:
                cost_m[ir, io] = min(cost_m[ir - 1, io] + ecl(r[ir], o[io]),
                                     cost_m[ir - 1, io - 1] + w * ecl(r[ir], o[io]),
                                     cost_m[ir, io - 1] + ecl(r[ir], o[io]))

    return cost_m[len(r) - 1, len(o) - 1] / (len(r) + len(o))


