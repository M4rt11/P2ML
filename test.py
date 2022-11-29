### fdp jupyter###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.ndimage import gaussian_filter1d
import os


## show all number of each subject

"""
for j in range(1,5):
    subject = 'Domain1_csv/Subject'+str(j)
    print(subject)
    for i in range(1, 11):
        print(subject + '-7-' + str(i) + '.csv')
        filename = subject + '-7-' + str(i) + '.csv'
        df = pd.read_csv(filename)
        plt.plot(df.iloc[3:, 0], df.iloc[3:, 1])
plt.show()
#plt.savefig('un.png')


##Matrix dtw
"""
for j in range(1, 11):
    subject = 'Domain1_csv/Subject'+str(j)
    print(subject)
    for i in range(1, 11):
        print(subject + '-8-' + str(i) +'.csv')
        filename = subject + '-8-' + str(i) +'.csv'
        df = pd.read_csv(filename)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        manhattan_distance = lambda x, y: np.abs(x - y)
        d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
plt.show()


for j in range(1,11):
    subject = 'Domain1_csv/Subject'+str(j)
    print(subject)
    for i in range(1, 5):
        print(subject + '-7-' + str(i) + '.csv')
        filename = subject + '-7-' + str(i) + '.csv'
        df = pd.read_csv(filename)
        x, y, z = np.array(df.iloc[3:, 0]), np.array(df.iloc[3:, 1]), np.array(df.iloc[3:, 2])
        x5 = gaussian_filter1d(x, sigma=5)
        y5 = gaussian_filter1d(y, sigma=5)
        z5 = gaussian_filter1d(z, sigma=5)
        manhattan_distance = lambda x5, y5: np.abs(x5 - y5)
        d, cost_matrix, acc_cost_matrix, path = dtw(x5, y5, dist=manhattan_distance)
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.grid()
plt.show()

