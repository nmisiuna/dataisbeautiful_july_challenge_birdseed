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
#print (data)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data)
#fig.colorbar(cax)
plt.yticks(range(len(index)), index, rotation = 45)
plt.xticks(range(len(data.keys())), data.keys(), rotation = 45)
plt.ylabel('Seed')
plt.ylabel('Bird')
plt.show(False)

#PCA
pca = PCA(n_components = len(data.keys()))
pca_model = pca.fit(data)
print (pca_model)
print (pca.explained_variance_ratio_)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show(False)

print (pca_model.components_[0])
plt.figure()
i = 0
for name in data.keys():
    if name == 'Purple Finches':
        print (i)
        print (pca_model.components_[0][i])
    if name == 'Goldfinches':
        print (i)
        print (pca_model.components_[0][i])
    plt.text(pca_model.components_[0][i], pca_model.components_[1][i], name)
    i += 1
plt.scatter(pca_model.components_[0], pca_model.components_[1])
plt.show(False)

#Let's look at a plot of just the 3 clusters, one by one in groups
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data[['Doves', 'Juncos']])
#fig.colorbar(cax)
plt.yticks(range(len(index)), index, rotation = 45)
plt.xticks(range(2), ('Doves', 'Juncos'), rotation = 45)
plt.ylabel('Seed')
plt.ylabel('Bird')
plt.show(False)

#Now Siskins + all finches
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data[['Siskins', 'House Finches', 'Goldfinches', 'Purple Finches']])
#fig.colorbar(cax)
plt.yticks(range(len(index)), index, rotation = 45)
plt.xticks(range(4), ('Siskins', 'House Finches', 'Goldfinches', 'Purple Finches'), rotation = 45)
plt.ylabel('Seed')
plt.ylabel('Bird')
plt.show(False)

#Now the rest
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data[['Sparrows', 'Titmice', 'Grosbeaks', 'Nuthatches', 'Towhees', 'Woodpeckers', 'Cardinals', 'Chickadees', 'Jays']])
#fig.colorbar(cax)
plt.yticks(range(len(index)), index, rotation = 45)
plt.xticks(range(9), ('Sparrows', 'Titmice', 'Grosbeaks', 'Nuthatches', 'Towhees', 'Woodpeckers', 'Cardinals', 'Chickadees', 'Jays'), rotation = 45)
plt.ylabel('Seed')
plt.ylabel('Bird')
plt.show(False)
