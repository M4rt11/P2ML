
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
            for coord_df in range(len(df)):
                coord.append([df.iloc[coord_df,0], df.iloc[coord_df,1]])
            #print(coord)

coord = pd.DataFrame(coord)
#print(coord.iloc[:,0])
#plt.scatter(coord.iloc[:, 0], coord.iloc[ :, 1], s = 50, c='b')
plt.show()

###CLUSTERING
kmeans = KMeans(init="random", n_init=10, max_iter=300, random_state=42, n_clusters=20)
kmeans.fit(coord.iloc[:, 0:2])
y_kmeans = kmeans.predict(coord.iloc[:, 0:2])
plt.scatter(coord.iloc[:, 0], coord.iloc[:,1], c=y_kmeans)
centers = kmeans.cluster_centers_
print(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
