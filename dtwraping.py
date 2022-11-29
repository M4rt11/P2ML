### DTW algorithm ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('Domain1_csv/Subject1-0-1.csv')
df2 = pd.read_csv('Domain1_csv/Subject1-0-2.csv')

print(df1.head())


x1 = np.array(df1.iloc[:20,0])
y1 = np.array(df1.iloc[:20,1])

x2 = np.array(df2.iloc[:,0])
y2 = np.array(df2.iloc[:,1])
'''

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()

from dtw import dtw

manhattan_distance = lambda x1, y1: np.abs(x1 - y1)

d, cost_matrix, acc_cost_matrix, path = dtw(x1, y1, dist=manhattan_distance)

print(d)

plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()

manhattan_distance = lambda x2, y2: np.abs(x2 - y2)

d, cost_matrix, acc_cost_matrix, path = dtw(x2, y2, dist=manhattan_distance)

print(d)

plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()
'''
### remplacer le module DTW par du code ##

## https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd

def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix

print(dtw(x1, x2))
print("----------------------------------------------------")


def dtw(s, t, window):
    n, m = len(s), len(t)
    w = np.max([window, abs(n - m)])
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            dtw_matrix[i, j] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            cost = abs(s[i - 1] - t[j - 1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix

print(dtw(x1, x2, window = 3))