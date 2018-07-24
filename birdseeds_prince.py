import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import prince

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

data.columns.rename('Bird Type', inplace = True)
data.index.rename('Seed Type', inplace = True)
data = data.transpose()

ca = prince.CA(n_components = len(data), n_iter = 3, copy = True, engine = 'auto')
#ca = prince.MCA(n_components = len(data), n_iter = 3, copy = True, engine = 'auto')
ca = ca.fit(data)

print (ca.column_principal_coordinates().head())

ca.plot_principal_coordinates()#show_column_labels = True, show_row_labels = True)
plt.show(False)

plt.figure()
plt.plot(np.cumsum(ca.explained_inertia_))
plt.plot(ca.explained_inertia_)
plt.show(False)
