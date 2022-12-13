import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from scipy import stats as st
from sklearn.cluster import KMeans


# STANDARDISATION
def standarization(data):
    scaler = StandardScaler()
    data.iloc[:, 0:2] = scaler.fit_transform(data.iloc[:, 0:2])  # que le x, y, z et pas le t
    return data


"""
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

                """
### Train and Test
coord_train = []
coord_test = []
###Train
x=5
for subject in range(1, 11):
    for digit in range(10):
        for rep in range(1, 11):
            if subject != x:
                f_name = f"Domain1_csv/Subject{subject}-{digit}-{rep}.csv"
                df = pd.read_csv(f_name)
                standarization(df)
                # print(df)
                numb = str(subject) + str(digit) + str(rep)
                for coord_df in range(len(df)):
                    coord_train.append([int(numb), df.iloc[coord_df, 0], df.iloc[coord_df, 1]])
            else:
                f_name = f"Domain1_csv/Subject{subject}-{digit}-{rep}.csv"
                df = pd.read_csv(f_name)
                standarization(df)
                # print(df)
                numb = str(subject) + str(digit) + str(rep)
                for coord_df in range(len(df)):
                    coord_test.append([int(numb), df.iloc[coord_df, 0], df.iloc[coord_df, 1]])

coord_train, coord_test = pd.DataFrame(coord_train), pd.DataFrame(coord_test)
print('train : ', coord_train)
#print('test : ', coord_test)
# print(coord.iloc[:,0])
# plt.scatter(coord.iloc[:, 0], coord.iloc[ :, 1], s = 50, c='b')
plt.show()

###créeation tableau pour mettre les sequence en relation avec les bon chiffre
data = np.zeros((900, 2))
for i in range(900):
    if i != x:
        data[i] = [int((i)// 100 + 1), int((i) // 10) % 10]
data_train = pd.DataFrame(data, columns=['UserID', 'Digit'])

data= np.zeros((100, 2))
for i in range(100):
    data[i] = [int((i)// 100 + 1), int((i) // 10) % 10]
data_test = pd.DataFrame(data, columns=['UserID', 'Digit'])


###CLUSTERING
kmeans = KMeans(init="random", n_init=10, max_iter=300, random_state=42, n_clusters=20)
kmeans.fit(coord_train.iloc[:, 1:3])
y_kmeans = kmeans.predict(coord_train.iloc[:, 1:3])
plt.scatter(coord_train.iloc[:, 1], coord_train.iloc[:, 2], c=y_kmeans)
centers = kmeans.cluster_centers_
print(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

coord_train["label"] = y_kmeans
### il faut attribuer les region du test en fonction des train, ici probleme :
#coord_test["label"] = y_kmeans



##renvoie les label de tt les coord du test en fct du kmeans fit avec le train
#print(kmeans.predict(coord_test.iloc[:, 1:3]))
coord_test["label"] = kmeans.predict(coord_test.iloc[:, 1:3])
#print(coord_test)

print(coord_train)
print(y_kmeans)

#print(coord_test[0])
#print(type(coord_train))

###Créer tableau avec les séquences des labels

def sequences(coord):
    ck = []
    string = []

    for i in range(len(coord)-1):
        if coord.iloc[i, 0] == coord.iloc[i + 1, 0]: #si la sequence d'apres est tjrs = a la sequence d'avant
            string.append(coord.iloc[i, -1]) #on ajoute la sequence a une liste
        else:
            ck.append(string)
            string = []
    return ck


sequences_train = sequences(coord_train)
print(len(sequences_train))
sequences_test = sequences(coord_test)
#print(sequences_train)

###Supprimer les repetitions
def supp_rep(ck):
    ck_new = []
    for i in range(len(ck)):
        newstring = []
        for indice in range(len(ck[i])):
            if indice == (len(ck[i]) - 1):
                newstring.append(ck[i][-1])
                # print('finish')
            elif ck[i][indice] != ck[i][indice + 1]:
                newstring.append(ck[i][indice])
            elif ck[i][indice] == ck[i][indice + 1]:
                pass
        ck_new.append(newstring)
    return ck_new


sequences_sansrep_train = supp_rep(sequences_train)
sequences_sansrep_test = supp_rep(sequences_test)
print(sequences_sansrep_train)
print(sequences_sansrep_test)


"""seq1 = [1, 1, 1, 1, 3, 3, 9]
seq2 = [1, 1, 0, 1, 3, 3, 9, 9]"""

# Definition edit distance
def edit_distance(stringtest, stringtrain):
    #print(stringtest, stringtrain)
    if len(stringtest) > len(stringtrain):
        difference = len(stringtest) - len(stringtrain)
        stringtest = stringtest[:-difference]

    if len(stringtrain) > len(stringtest):
        #print('ok')
        difference = len(stringtrain) - len(stringtest)
        stringtrain = stringtrain[:-difference]

    else:
        difference = 0
    #print(stringtest, stringtrain)
    for i in range(len(stringtest)-1):
        #print(len(stringtest))
        #print(i)
        if stringtest[i] != stringtrain[i]:
            difference += 1

    return difference


##print('diff =', edit_distance(seq1, seq2))

##boucle qui compare chaque seq de test a toute les sequence de train :
"""
sequences_sansrep_test = [[1, 2, 1, 2, 3, 4, 9, 8],
                          [1, 1, 1, 2, 3, 4, 9]]

sequences_sansrep_train = [[1, 1, 1, 1, 3, 3, 9],
                          [1, 2, 1, 2, 3, 4, 9, 8],
                           [1, 2, 4, 2, 3, 4, 5, 8]
                           ]"""
print(len(sequences_sansrep_test))
ok = 0
for seqtest in range(len(sequences_sansrep_test)):
    meilleur = 10000
    number_of_seq_train = 0
    for seqtrain in range(len(sequences_sansrep_train)):
        diff = edit_distance(sequences_sansrep_test[seqtest], sequences_sansrep_train[seqtrain])
        #print('diff = ', diff)
        if diff < meilleur:
            meilleur = diff
            number_of_seq_train = seqtrain
            #print('meilleiur' ,meilleur, number_of_seq_train)
        else:
            pass
    #print(meilleur, number_of_seq_train, seqtest)
    #print("Test seq : ",seqtest, "on trouve : ", number_of_seq_train, "avec une diff de ", meilleur)
    print(f"quand {data_test.iloc[seqtest,1]} : model predit {data_train.iloc[number_of_seq_train, 1]} with a difference of {meilleur}")
    if data_test.iloc[seqtest,1] == data_train.iloc[number_of_seq_train, 1]:
        ok += 1
    else:
        pass
    print('accuracy:',  ok/100)







