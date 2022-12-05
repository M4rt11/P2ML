### fdp jupyter###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

from scipy import stats as st

import os

import time
from tqdm import tqdm
def standarization(data):
    scaler = StandardScaler()
    data.iloc[:, 0:2] = scaler.fit_transform(data.iloc[:, 0:2])
    return data

## show all number of each subject

for j in range(1,11):
    subject = 'Domain1_csv/Subject'+str(j)
    print(subject)
    for i in range(1, 11):
        print(subject + '-5-' + str(i) + '.csv')
        filename = subject + '-5-' + str(i) + '.csv'
        df = pd.read_csv(filename)
        df = standarization(df)
        plt.plot(df.iloc[:, 0], df.iloc[:, 1])
plt.show()

for j in range(1,11):
    subject = 'Domain1_csv/Subject'+str(j)
    print(subject)
    for i in range(1, 5):
        print(subject + '-5-' + str(i) + '.csv')
        filename = subject + '-5-' + str(i) + '.csv'
        df = pd.read_csv(filename)
        df = standarization(df)
        x, y, z= np.array(df.iloc[3:, 0]), np.array(df.iloc[3:, 1]), np.array(df.iloc[3:, 2])
        x5 = gaussian_filter1d(x, sigma=5)
        y5 = gaussian_filter1d(y, sigma=5)
        z5 = gaussian_filter1d(z, sigma=5)
        #plt.plot(x, y, 'k', label='original data')
        plt.plot(x5,y5, label='filtered, sigma=5')
        plt.legend()
        plt.grid()
plt.show()
"""
##Matrix dtw

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
#plt.show()

###
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
        print(d, cost_matrix, acc_cost_matrix, path)
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.grid()
#plt.show()



"""
data = np.zeros((1000, 2))
for i in range(1000):
    data[i] = [ int((i) // 100 + 1), int((i) // 10) % 10]
data = pd.DataFrame(data, columns=['UserID', 'Digit'])


coord = []
for subject in range(1,11):
    for digit in range(10):
        for rep in range(1, 11):
            f_name = f"Domain1_csv/Subject{subject}-{digit}-{rep}.csv"
            df = pd.read_csv(f_name)
            standarization(df)
            x, y, z = np.array(df.iloc[3:, 0]), np.array(df.iloc[3:, 1]), np.array(df.iloc[3:, 2])
            x5 = gaussian_filter1d(x, sigma=5)
            y5 = gaussian_filter1d(y, sigma=5)
            z5 = gaussian_filter1d(z, sigma=5)
            coord.append(np.array([x5, y5]))
coord = pd.Series(coord)
data["Coord"] = coord

df_user = data[data["UserID"]==4]
coord_digit = df_user[df_user['Digit']==1]
#print(coord_digit['Coord'])
#print(coord_digit['Coord'].iloc[1])


print(dtw(coord_digit['Coord'].iloc[1],coord_digit['Coord'].iloc[9]))



###Cross validation ###
k = 15

Train = data[data['UserID'] != 3]
Test = data[data['UserID'] == 3]

tik = time.perf_counter()
for te in range(0,100):
    signal = Test.iloc[te, 2]
    #print(signal)
    result = []
    result_d = []
    for tr in range(0,900):
        r = Train.iloc[tr,2]
        result.append(dtw(signal, r))
        result_d.append(Train.iloc[tr, 1])

    order = np.array(result_d)[np.argsort(np.array(result))][:k]
    predicted = st.mode(order)[0]
    if te % 10 ==0:
        print(f'True = {Test.iloc[te, 1]}  - Pred = {predicted}')




print(f'Time to execute : {time.perf_counter()-tik:.3f} sec.')




