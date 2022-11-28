### DTW algorithm ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('Domain1_csv/Subject1-0-1.csv')
df2 = pd.read_csv('Domain1_csv/Subject1-0-2.csv')

print(df1.head())


x1 = np.array(df1.iloc[:,0])
y1 = np.array(df1.iloc[:,1])

x2 = np.array(df2.iloc[:,0])
y2 = np.array(df2.iloc[:,1])


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
