
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

#CREATION DU TABLEAU

  # On crée une matrice de 0 (2 colonnes et 1000 lignes)
data = np.zeros((1000, 2)) # (Colonnes : 1 = subject, 2 = number qu'il fait)
for i in range(1000):
    data[i] = [int((i) // 100 + 1), int((i) // 10) % 10]
data = pd.DataFrame(data, columns=['UserID', 'Digit'])



 #On crée une liste de coordonnées x et y pour chaque répétition, liste coord [[x],[y]]
coord = []
for subject in range(1,11):
    for digit in range(10):
        for rep in range(1, 11):
            f_name = f"Domain1_csv/Subject{subject}-{digit}-{rep}.csv"
            df = pd.read_csv(f_name)
            standarization(df)
            #print(df)
            for coord_df in range(len(df)):
                coord.append([df.iloc[coord_df,0], df.iloc[coord_df,1]])
            #print(coord)


coord = pd.DataFrame(coord)
#print(coord.iloc[:,0])
#plt.scatter(coord.iloc[:, 0], coord.iloc[ :, 1], s = 50, c='b')
plt.show()

kmeans = KMeans(init="random", n_init=10, max_iter=300, random_state=42, n_clusters=20)
kmeans.fit(coord.iloc[:, 0:2])
y_kmeans = kmeans.predict(coord.iloc[:, 0:2])
plt.scatter(coord.iloc[:, 0], coord.iloc[:,1], c=y_kmeans)
centers = kmeans.cluster_centers_
print(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
facet = sns.lmplot(data=data, x='x', y='y', hue='label',
                   fit_reg=False, legend=True, legend_out=True)

"""           
#coord = pd.Series(coord) #on dit que les coord sont égales à un dataframe
data["Coord"] = coord #on crée un nouvel index coord dans la matrice data, on rajoute une colonne

#print(data["Coord"])


df = pd.read_csv('Domain1_csv/Subject1-0-1.csv')
df2 = pd.read_csv('Domain1_csv/Subject1-7-2.csv')
kmeans = KMeans(init="random", n_init=20, max_iter=300, random_state=42)
kmeans.fit(df.iloc[:, 0:2])
kmeans.predict(df2.iloc[:,0:2])
print(kmeans.score(df2.iloc[:,:2]))

X = df.iloc[:, :2]
X1 = df2.iloc[:, :2]


#print(X.iloc[:,0], X.iloc[:,1])

plt.scatter(X.iloc[:, 0], X.iloc[ :, 1], s = 50, c='b')
plt.show()
Kmean = KMeans(n_clusters=20)
Kmean.fit(X)

print(Kmean.cluster_centers_)
plt.scatter(X.iloc[:, 0], X.iloc[ :, 1], s = 50, c='b')
plt.scatter(0.0531858,  -0.0566656 , s=200, c='g', marker='s')
plt.scatter(0.00407643,  0.04564843, s=200, c='r', marker='s')
plt.show()
"""

