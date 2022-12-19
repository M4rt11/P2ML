import pandas as pd
import numpy as np


###_______Distance Time Warping_______###

def ecl(train, test): #distance euclidienne formule

    return ((train[0] - test[0]) ** 2 + (train[1] - test[1]) ** 2) ** 0.5

def dtw(train, test, w=2):

    #w: weight for diagonal transition, we fixed to 2

    ## Creation of the cost matrix
    cost_m = np.zeros((len(train), len(test)))  # Create the cost matrix, prend le taille du train et du test

    for itrain in range(len(train)): #algo qui vient des slides du cours
        for itest in range(len(test)):
            # First: Origin
            if itest == itrain == 0:
                cost_m[0, 0] = ecl(train[itrain], test[itest])
            # Then let's continue with the first row or first column
            elif itrain == 0:
                cost_m[itrain, itest] = cost_m[itrain, itest - 1] + ecl(train[itrain], test[itest])
            elif itest == 0:
                cost_m[itrain, itest] = cost_m[itrain - 1, itest] + ecl(train[itrain], test[itest])
            # Finish with the rest of the matrix
            else:
                cost_m[itrain, itest] = min(cost_m[itrain - 1, itest] + ecl(train[itrain], test[itest]),
                                     cost_m[itrain - 1, itest - 1] + w * ecl(train[itrain], test[itest]),
                                     cost_m[itrain, itest - 1] + ecl(train[itrain], test[itest]))

    return cost_m[len(train) - 1, len(test) - 1] / (len(train) + len(test)) #ca nous retourne le cout min entre les test et le train normalisé sur la longueur car le cout dépend aussi de la longueur


