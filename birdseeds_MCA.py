import pandas as pd
import matplotlib.pyplot as plt
import mca
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = {'Cardinals': [3, 3, 3, 1, 0, 0, 0, 3, 1],
        'Chickadees': [3, 2, 2, 0, 0, 0, 2, 2, 0],
        'Doves': [2, 1, 2, 3, 3, 1, 0, 2, 2],
        'Goldfinches': [3, 2, 3, 1, 0, 3, 0, 0, 0],
        'Grosbeaks': [2, 2, 3, 0, 0, 0, 0, 1, 0],
        'House Finches': [3, 2, 3, 2, 0, 3, 0, 1, 0],
        'Jays': [3, 3, 3, 0, 1, 0, 2, 1, 2],
        'Juncos': [1, 1, 1, 1, 0, 1, 0, 0, 3],
        'Nuthatches': [3, 2, 2, 0, 0, 0, 1, 1, 0],
        'Purple Finches': [3, 2, 3, 1, 0, 3, 0, 0, 0],
        'Siskins': [1, 1, 3, 0, 0, 3, 0, 0, 1],
        'Sparrows': [3, 3, 3, 3, 2, 0, 0, 1, 2],
        'Titmice': [3, 2, 2, 0, 0, 1, 2, 1, 0],
        'Towhees': [3, 3, 3, 1, 0, 0, 1, 1, 1, ],
        'Woodpeckers': [3, 3, 2, 0, 0, 0, 1, 1, 1,]}
index = ('Black Oil Sunflower', 'Striped Sunflower', 'Hulled Sunflower', 'Millet White/Red', 'Milo Seed', 'Nyjer Seed (Thistle)', 'Shelled Peanuts', 'Safflower Seed', 'Corn Products')
data = pd.DataFrame(data = data, index = index)

#EXPLAINED VARIANCE DOES NOT SUM TO ONE
#data = data.transpose()
#print ("dummies\n")
#print (pd.get_dummies(data, columns = list(data)))
data_dummies = pd.get_dummies(data, columns = list(data))

mca_ben = mca.MCA(data_dummies, ncols = len(data_dummies.keys()))
#print (mca_ben.fs_r(1))
#print (np.cumsum(mca_ben.expl_var()))
#print (mca_ben.L)
#print (mca_ben.inertia)

print (len(mca_ben.fs_r(1)[1]))

plt.figure()
i = 0
for name in list(data.index):
    plt.text(mca_ben.fs_r(1)[0][i], mca_ben.fs_r(1)[1][i], name)
    i += 1
plt.scatter(mca_ben.fs_r(1)[0], mca_ben.fs_r(1)[1])
plt.show(False)

#Explained variance
N_eig_all = len(data_dummies.keys())
expl_var_bn = []
#print (np.sum(mca_ben.expl_var(N = len(data.keys()))))
for N_eig in range(0, N_eig_all):
    expl_var_bn.append(np.sum(mca_ben.expl_var(N = N_eig)))

#plt.figure()
#plt.plot(range(0, N_eig_all), expl_var_bn)
#plt.show(False)

