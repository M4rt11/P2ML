# Imports
from dtw import *
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import statistics


def norm(v):
    def norm(v):
        #v = (v - np.min(v)) / (np.max(v) - np.min(v))
        v = (v - statistics.mean(v)/ statistics.stdev(v))
        return v

### DYNAMIC TIME WRAPPING METHOD
def standarization(data):
    scaler = StandardScaler()
    data.iloc[:, 0:2] = scaler.fit_transform(data.iloc[:, 0:2]) #que le x, y, z et pas le t
    return data

#CREATION DU TABLEAU
  # On crée une matrice de 0 (2 colonnes et 1000 lignes)
data = np.zeros((1000, 2)) # (Colonnes : 1 = subject, 2 = number qu'il fait)
for i in range(1000):
    data[i] = [ int((i) // 100 + 1), int((i) // 10) % 10]
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
data["Coord"] = coord


####  Dynamic time wrapping
""" HOW DO I PLAN IT:
We will compare a test digit with all training digits of each class.
    The test digit will be classified in the class were the mean cumulative difference is the lowest.
"""

# Optimal cumulative difference calculation function
def dtw(r, o, w=2, normal=False):
    """
    :param r: num of the reference signal's file => a 2 dimensions  x y vector
    :param o: nom of the observed signal's file => a 2 dimensions x y  vector
    :param w: weight for diagonal transition, in our project it's fixed to 2
    :param normal: If True, the inputs vector are normalized btw 0 and 1
    :return: scalar = to the lowest cumulative difference between the two signals
    """
    ## Creation of the cost matrix
    cost_m = np.zeros((len(r), len(o)))  # Create the cost matrix

    for ir in range(len(r)):
        for io in range(len(o)):
            # First: Origin
            if io == ir == 0:
                cost_m[0, 0] = ecl(r[ir], o[io])
            # Then let's continue with the first row or first column
            elif ir == 0:
                cost_m[ir, io] = cost_m[ir, io - 1] + ecl(r[ir], o[io])
            elif io == 0:
                cost_m[ir, io] = cost_m[ir - 1, io] + ecl(r[ir], o[io])
            # Finish with the rest of the matrix
            else:
                cost_m[ir, io] = min(cost_m[ir - 1, io] + ecl(r[ir], o[io]),
                                     cost_m[ir - 1, io - 1] + w * ecl(r[ir], o[io]),
                                     cost_m[ir, io - 1] + ecl(r[ir], o[io]))

    return cost_m[len(r) - 1, len(o) - 1] / (len(r) + len(o))  # Normalize the results due to different lengths

# Dataset split
train = data[data["UserID"] != 10].reset_index()  # 9 first users = train useres
test = data[data["UserID"] == 10].reset_index()  # Last user = test user
i = 0
total = 0
# Classification of test the digit
for t in range(23, len(test)):
    print("t = ", t)
    pred_d = 99  # Initial value for digit prediction, should be changed anyway in the loop
    min_dtw = 999  # Initial value for min cumulated difference, should be changed anyway in the loop

    for d in range(10):  # We will compare it to the 10 digit class (0-9)
        sub_digit = train[train["Digit"] == d].reset_index()  # Selection every digit class one at time
        score = []  # Attribute a score for every digit class
        #print("Comparing {0} with digit {1}".format(test["Digit"][t], d))

        for j in range(len(sub_digit)):  # We compare our test obs with all the selected train obs
            score.append(dtw(sub_digit["Coord"][j],
                             test["Coord"][t],
                             normal=True))  # We attribute a score for every train obs with the test obs. We sum all the score for the same digit

        m_score = sum(score) / len(score)  # We compute the mean for that digit
        if m_score < min_dtw:  # If the mean score is lower
            min_dtw = m_score  # then it becomes the min_score to beat for future digits
            #print("{0} -> {1}".format(pred_d, d))
            pred_d = d  # We then predict the associated the digit



    total += 1
    print("True D: {0} - Pred D: {1}".format(test["Digit"][t], pred_d))
    if test["Digit"][t] == pred_d:
        i += 1
print("accuracy :", i/total)

""" RESULTS:
+ Always correct except for 0 which is always wrong
- Takes a lot of time because we have to compare every test digit with the 900 train digits.

"""

