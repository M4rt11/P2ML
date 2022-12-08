

""""
for two strings
    x of len n
    y of len m

d(n,m) is edit distance between x[1..n] and y[1...m]
"""

def editdist (n,m):

    #Initialization

    d(n,0) = n
    d(0,m) = m


    for i in len(y):
        for j in len(x):
            if x(n) != y(m):
                z = 2
            else:
                z = 0

            d(n, m) = min(d(n - 1, m) + 1,
                          d(n, m - 1) + 1,
                          d(n - 1, m - 1) + z)