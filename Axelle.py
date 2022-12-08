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
Coord = np.zeros((1000, 2))
for subject in range(1,11):
    for digit in range(10):
        for rep in range(1, 11):
            f_name = f"Domain1_csv/Subject{subject}-{digit}-{rep}.csv"
            df = pd.read_csv(f_name)
            standarization(df)



