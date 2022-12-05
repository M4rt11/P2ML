
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
            x, y, z = np.array(df.iloc[3:, 0]), np.array(df.iloc[3:, 1]), np.array(df.iloc[3:, 2])
            x5 = gaussian_filter1d(x, sigma=5) #on filtre les points
            y5 = gaussian_filter1d(y, sigma=5)
            z5 = gaussian_filter1d(z, sigma=5)
            coord.append(np.array([x5, y5]))
coord = pd.Series(coord) #on dit que les coord sont égales à un dataframe
data["Coord"] = coord #on crée un nouvel index coord dans la matrice data, on rajoute une colonne


kmeans = KMeans(init="random", n_init=10, max_iter=300, random_state=42)
kmeans.fit(data["Coord"])