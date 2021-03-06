import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import prince
from sklearn.cluster import KMeans
from adjustText import adjust_text
import matplotlib.gridspec as gridspec

pic_type = 'png'

data = pd.read_csv('./birdseed.csv', sep = ',', header = 0, index_col = 0)

data.columns.rename('Bird Type', inplace = True)
data.index.rename('Seed Type', inplace = True)

#Correspondence Analysis using the prince library
ca = prince.CA(n_components = len(data), n_iter = 3, copy = True, engine = 'auto')
ca = ca.fit(data)

#Plot the inertia to justify the use of this method
plt.figure()
plt.plot(np.cumsum(ca.explained_inertia_), label = 'Ind. Inertia')
plt.plot(ca.explained_inertia_, label = 'Cum. Sum. of Inertia')
plt.legend(loc = 2, fancybox = True, framealpha = 1)
plt.title('Correspondence Analysis Inertia')
plt.xlabel('Principal Components')
plt.ylabel('Inertia')
plt.savefig('CA Inertia.png', format = 'png', bbox_inches = 'tight')
plt.show(False)

#Let's try to apply k-means algorithm to the CA data to auto-identify clusters
n_clusters = 3
kmeans = KMeans(n_clusters = n_clusters)
kmeans = kmeans.fit(ca.column_coordinates(data)[[0, 1]])
labels = kmeans.predict(ca.column_coordinates(data)[[0, 1]])
centroids = kmeans.cluster_centers_

#Make labels for the transposed variables so we can do full segmentation
labels_T = kmeans.predict(ca.row_coordinates(data)[[0, 1]])

ax = ca.plot_coordinates(data, show_col_labels = False, show_row_labels = False)
ax.get_figure().set_size_inches(9, 9, forward = True)
#I need to construct my own set of text labels and use the adjustText library
#to fix the overlapping text issue
row_labels = ca.row_coordinates(data)
full_texts = zip(row_labels[0], row_labels[1], data.index)
texts = []
for x, y, name in full_texts:
    texts.append(plt.text(x, y, name, fontsize = 12))
col_labels = ca.column_coordinates(data)
full_texts = zip(col_labels[0], col_labels[1], data.columns)
for x, y, name in full_texts:
    texts.append(plt.text(x, y, name, fontsize = 12))
adjust_text(texts)

#Plot the centers of the k-means centroids and annotate them            
plt.scatter(centroids[:,0], centroids[:,1])
for i in range(0, n_clusters):
    plt.annotate('%s' % i, centroids[i],  bbox=dict(boxstyle='circle', fc = 'none', 
                 ec = 'black', alpha = 0.8), size = 15, color = 'black')
plt.legend(loc = 1, fancybox = True, framealpha = 1)

plt.savefig('CA Principle Components.png', format = 'png', bbox_inches = 'tight')
plt.show(False)

#Now I know which seeds go in which cluster
#Let's make an automated heat map of seed/bird clusters
fig = plt.figure()
fig.set_size_inches(6, len(data) / len(list(data)) * 6.0, forward = True)

#Now make the n_cluster number of subplots
#I want to order them in decreasing size (number of elements per cluster)
#Find the biggest cluster first
unique, counts = np.unique(labels, return_counts = True)
labeldict = dict(zip(counts, unique))

#Do a sort on counts and apply the same sorting to unique
#I'll keysort counts and heat map plot each bird or seed in that cluster

width_ratios = sorted(counts, reverse = True)
width_ratios.append(1)
grid = gridspec.GridSpec(1, n_clusters + 1, width_ratios = width_ratios)

iteration = 0
total = 0
for cluster in sorted(labeldict, reverse = True):
    i = labeldict[cluster]
    z = [] #The cluster's data
    for j in range(0, len(data.columns)):
        if (labels[j] == i):
            z.append(j)
    #ax = fig.add_subplot(1, n_clusters + 1, iteration + 1)
    ax = fig.add_subplot(grid[iteration])#[:, iteration])
    total += cluster
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    cax = ax.matshow(data[data.columns[z[:]]], aspect = "auto")#aspect = "equal")
    plt.xticks(range(len(z)), list(data[data.columns[z[:]]]), rotation = 30, ha = 'left')
    if(iteration == 0):
        plt.yticks(range(len(data.index)), data.index, rotation = 30)
    else:
        #I want a shared y-axis
        plt.yticks([], [])
    iteration += 1

    #Add white lines separating each value via gridlines
    #The gridlines go off the minor axes
    #Also, don't overwrite the boundary of the plot with white lines
    ax.set_yticks(np.arange(0.5, len(data.index) - 1, 1), minor = True)
    ax.set_xticks(np.arange(0.5, len(z) - 1, 1), minor = True)
    ax.grid(which = 'minor', axis = 'both', color = 'w', linestyle = '-', linewidth = 2)

#Let's make our own legend as a subplot
ax = fig.add_subplot(grid[n_clusters])
m = np.zeros((1, 4))
for i in range(4):
    m[0, i] = 100.0 - (i * 4) / 100.0 #No idea why it's (i * 4) / 100
plt.imshow(np.transpose(m), aspect = "auto")
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.yaxis.tick_right()
plt.yticks(range(4), range(3, -1, -1))
plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
plt.title('Legend', fontsize = 12)

#Adjust the subplot separations
plt.savefig('%s Clusters Matrix.%s' % (data.columns.name, pic_type), format = 'png', bbox_inches = 'tight')
plt.show(False)    

#Now let's make a full segmented subplot for clusters in both axes
#This is n_cluster by n_cluster, which is bad because one cluster might have
#zero of one dimension or dimension_T in it
#This is therefore not the generalized implementation

#Get the cluster data for the transposed data
unique_T, counts_T = np.unique(labels_T, return_counts = True)
labeldict_T = dict(zip(counts_T, unique_T))

#I need to get appropriate height ratios based on the sorting of the columns
#This keeps clusters together in both row/columns rather than independent
z = dict(zip(unique_T, counts_T)) #This works but it feels sloppy
height_ratios = []
for i in range(0, n_clusters):
    height_ratios.append(z[labeldict[sorted(labeldict, reverse = True)[i]]])

width_ratios = sorted(counts, reverse = True)
width_ratios.append(1)

grid = gridspec.GridSpec(n_clusters, n_clusters + 1, width_ratios = width_ratios, height_ratios = height_ratios)

fig = plt.figure()
fig.set_size_inches(6, len(data) / len(list(data)) * 6.0, forward = True)

#Set up my set of axes
ax = [[] for i in range(0, n_clusters * n_clusters)]

iteration = 0 #Tracking subplots
iteration_total = 0 #Tracking subplots
for cluster in sorted(labeldict, reverse = True): #First do columns
    i = labeldict[cluster]
    z = [] #The cluster's data
    for j in range(0, len(data.columns)): #Grabbing columns in the cluster
        if (labels[j] == i):
            z.append(j)
            
    iteration_T = 0
    for cluster_T in sorted(labeldict, reverse = True): #Then rows
        i_T = labeldict[cluster_T]
        z_T = [] #The cluster's transposed dimension data
        for j_T in range(0, len(data.index)): #Grabbing rows in the cluster
            if (labels_T[j_T] == i_T):
                z_T.append(j_T)

        #Turn these into a subplot
        ax[iteration_total] = fig.add_subplot(grid[iteration_T, iteration])
        ax[iteration_total].tick_params(axis = 'both', which = 'major', labelsize = 12)
        cax = ax[iteration_total].matshow(data.iloc[z_T[:], z[:]], aspect = "auto", vmin = data.min().min(), vmax = data.max().max())#aspect = "equal")
        if(iteration_T == 0):
            plt.xticks(range(len(z)), list(data[data.columns[z[:]]]), rotation = 30, ha = 'left')
        else:
            #I want a shared x-axis
            plt.xticks([], [])
        if(iteration == 0):
            plt.yticks(range(len(data.index[z_T[:]])), data.index[z_T[:]], rotation = 30)
        else:
            #I want a shared y-axis
            plt.yticks([], [])

        #Add white lines separating each value via gridlines
        #The gridlines go off the minor axes
        #Also, don't overwrite the boundary of the plot with white lines
        ax[iteration_total].set_yticks(np.arange(0.5, len(data.index[z_T[:]]) - 1, 1), minor = True)
        ax[iteration_total].set_xticks(np.arange(0.5, len(z[:]) - 1, 1), minor = True)
        ax[iteration_total].grid(which = 'minor', axis = 'both', color = 'w', linestyle = '-', linewidth = 2)
        
        iteration_T += 1
        iteration_total += 1
    iteration += 1

#Let's make our own legend as a subplot
ax = fig.add_subplot(grid[:, n_clusters])
m = np.zeros((1, 4))
for i in range(4):
    m[0, i] = 100.0 - (i * 4) / 100.0 #No idea why it's (i * 4) / 100
plt.imshow(np.transpose(m), aspect = "auto")
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.yaxis.tick_right()
plt.yticks(range(4), range(3, -1, -1))
plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
plt.title('Legend', fontsize = 12)

plt.savefig('Segmented Clusters Matrix.%s' % pic_type, format = 'png', bbox_inches = 'tight')
plt.show(False)  
