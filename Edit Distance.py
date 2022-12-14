
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from scipy import stats as st
from sklearn.cluster import KMeans

#STANDARDISATION
def standarization(data):
    scaler = StandardScaler()
    data.iloc[:, 0:2] = scaler.fit_transform(data.iloc[:, 0:2]) #que le x, y, z et pas le t
    return data


### liste de coordonées de point x, y
coord = []
for subject in range(1,11):
    for digit in range(10):
        for rep in range(1, 11):
            f_name = f"Domain1_csv/Subject{subject}-{digit}-{rep}.csv"
            df = pd.read_csv(f_name)
            standarization(df)
            #print(df)
            numb = str(subject) + str(digit) + str(rep)
            for coord_df in range(len(df)):
                coord.append([int(numb),df.iloc[coord_df,0], df.iloc[coord_df,1]])


coord = pd.DataFrame(coord)
#print(coord.iloc[:,0])
#plt.scatter(coord.iloc[:, 0], coord.iloc[ :, 1], s = 50, c='b')
plt.show()

###CLUSTERING
kmeans = KMeans(init="random", n_init=10, max_iter=300, random_state=42, n_clusters=20)
kmeans.fit(coord.iloc[:, 1:3])
y_kmeans = kmeans.predict(coord.iloc[:, 1:3])
plt.scatter(coord.iloc[:, 1], coord.iloc[:, 2], c=y_kmeans)
centers = kmeans.cluster_centers_
print(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

coord["label"] = y_kmeans
print(coord[0])
print(type(coord))

###Créer tableau avec les séquences des labels

ck = []
string=[]

for i in range(len(coord)-1):
    if coord.iloc[i,0] == coord.iloc[i+1,0]:
        string.append(coord.iloc[i,-1])
    else:
        ck.append(string)
        string = []

print(ck)



###Supprimer les repetitions
ck_new = []
for i in range(len(ck)):
    newstring=[]
    for indice in range(len(ck[i])):
        if indice == len(ck[i])-1:
            newstring.append(ck[i][-1])
            #print('finish')
        elif ck[i][indice] != ck[i][indice+1]:
            newstring.append(ck[i][indice])
        elif ck[i][indice] == ck[i][indice+1]:
               pass
    ck_new.append(newstring)

print(ck_new)
print(type(ck_new))


"""
#Definition edit distance
def edit_distance(stringtest, stringtrain):

    if len(stringtest) > len(stringtrain):
        difference = len(stringtest) - len(stringtrain)
        stringtest = stringtest[:difference]

    if len(stringtrain) > len(stringtest):
        difference = len(stringtrain) - len(stringtest)
        stringtrain = stringtrain[:difference]

    else:
        difference = 0

    for i in range(len(stringtest)):
        if stringtest[i] != stringtrain[i]:
            difference += 1

    return difference

print(edit_distance(ck_new[0], ck_new[32]))

"""
#Retourner la difference minimale