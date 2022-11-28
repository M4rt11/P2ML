### fdp jupyter###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
import os


## show all number of each subject


for j in range(1,11):
    subject = 'Domain1_csv/Subject'+str(j)
    print(subject)
    for i in range(1, 11):
        print(subject + '-0-'+ str(i) +'.csv')
        filename = subject + '-0-' + str(i) +'.csv'
        df = pd.read_csv(filename)
        plt.plot(df.iloc[:, 0], df.iloc[:, 1])
plt.show()
#plt.savefig('un.png')

'''
##Matrix dtw

for j in range(1, 11):
    subject = 'Domain1_csv/Subject'+str(j)
    print(subject)
    for i in range(1, 11):
        print(subject + '-0-' + str(i) +'.csv')
        filename = subject + '-0-' + str(i) +'.csv'
        df = pd.read_csv(filename)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        manhattan_distance = lambda x, y: np.abs(x - y)
        d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
plt.show()

for j in range(1,5):
    subject = 'Domain1_csv/Subject'+str(j)
    print(subject)
    for i in range(1, 11):
        print(subject + '-0-' + str(i) +'.csv')
        filename = subject + '-0-' + str(i) + '.csv'
        df = pd.read_csv(filename)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

df = pd.read_csv('Domain1_csv/Subject4-0-10.csv')

text = open("Domain1_csv/Subject4-0-10.csv", "r")

# join() method combines all contents of
# csvfile.csv and formed as a string
text = ''.join([i for i in text])

# search and replace the contents
text = text.replace(";", ",")
'''

