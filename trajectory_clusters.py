# trajectory clusters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
# # for loading the trajectories
workpath = '/glade/work/psturm/ice-mp-su24/'
os.chdir(workpath)

# seed for reproducibility
np.random.seed(42)


# load data
rh_trajectories = pd.read_csv('saved_trajectory_data/rh_trajectories.csv')
rh_trajectories = rh_trajectories.dropna()

# make the index rk_deact
rh_trajectories = rh_trajectories.set_index('rk_deact')

# convert to numpy array
supersaturation = (rh_trajectories.to_numpy()-1)

# histogram of supersaturation
plt.hist(supersaturation.flatten(), bins=100)
# yaxis label is superdroplet count (not multiplicity weighted)
plt.ylabel('superdroplet count (not $\\xi$ weighted)')
plt.xlabel('supersaturation with respect to ice [%]')



# make an elbow plot to determine the number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(init="k-means++", n_clusters=k, n_init=4)
    kmeans.fit(supersaturation)
    sse.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), sse)
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Distances from Centroids")
plt.show()


kmeans = KMeans(init="k-means++", n_clusters=5, n_init=4)
kmeans.fit(supersaturation)

# plot the cluster centers as a timeseries
plt.figure()

# color code the clusters with a scientific colormap
colors = plt.cm.plasma(np.linspace(0, 1, kmeans.cluster_centers_.shape[0]))
markers = ['o', 's', 'v', 'x', 'd']
# get the standard deviation of each cluster at each time step
stds = np.array([np.std(supersaturation[kmeans.labels_ == i], axis=0) for i in range(kmeans.cluster_centers_.shape[0])])


# plot the cluster centers with the standard deviation as ribbons


# Plot the cluster centers with the standard deviation as ribbons
for i in range(kmeans.cluster_centers_.shape[0]):
    center = kmeans.cluster_centers_[i,:].T * 100
    std_dev = stds[i,:] * 100
    time = np.arange(center.shape[0])  # Assuming time is a sequence of integers

    plt.plot(time, center, color=colors[i % len(colors)], label='cluster {}'.format(i+1),
             marker=markers[i % len(markers)], linestyle='-', markersize=4)
    plt.fill_between(time, center - std_dev, center + std_dev, color=colors[i % len(colors)], alpha=0.2)

plt.xlabel('Time [minutes]')
plt.ylabel('Supersaturation with respect to ice [%]')
plt.legend()
plt.show()


reduced_data = PCA(n_components=2).fit_transform(supersaturation)
kmeans = KMeans(init="k-means++", n_clusters=5, n_init=4)
kmeans.fit(reduced_data)

# Get the explained variance ratio of the principal components
explained_variance_ratio = PCA(n_components=2).fit(supersaturation).explained_variance_ratio_


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=   plt.cm.plasma,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "bx", markersize=2)
# Get the percentage of points in each cluster
cluster_sizes = np.bincount(kmeans.labels_)/len(kmeans.labels_)*100
for i in range(kmeans.cluster_centers_.shape[0]):
    text = plt.text(
        kmeans.cluster_centers_[i, 0],
        kmeans.cluster_centers_[i, 1],
        str(cluster_sizes[i].round(1)) + "%",
        color="white",
        fontsize=7,
        fontweight="bold",
    )
    text.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])

plt.title(
    "Trajectory clusters in the first two principal components\n"
    "Total explained variance ratio: {:.1f}".format(np.sum(explained_variance_ratio)*100) + "%"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# label PCA axes
plt.xlabel("1st principal component, explained variance ratio: {:.1f}".format(explained_variance_ratio[0]*100) + "%")
plt.ylabel("2nd principal component, explained variance ratio: {:.1f}".format(explained_variance_ratio[1]*100) + "%")
plt.xticks(())
plt.yticks(())
plt.show()

# load trajs and add the cluster labels to the dataframe
trajs = pd.read_csv('saved_trajectory_data/trajs_5100_7200_rand1000sds.csv')
# add the cluster labels to the dataframe
rh_trajectories['cluster'] = kmeans.labels_+1

# left join the cluster labels to the trajs dataframe
trajs = trajs.merge(rh_trajectories['cluster'], left_on='rk_deact', right_index=True)
# save the trajs dataframe with the cluster labels
trajs.to_csv('saved_trajectory_data/trajs_5100_7200_rand1000sds_clusters.csv', index=False)


def plot_cluster_means_with_std_dev(trajs, column_name, scale_factor=1, ylabel=None):
    plt.figure()
    # get the standard deviation of each cluster at each time step
    data = trajs.pivot(index='rk_deact', columns='time', values=column_name)
    # convert to numpy array
    data = data.to_numpy() * scale_factor
    stds = np.array([np.std(data[kmeans.labels_ == i], axis=0) for i in range(kmeans.cluster_centers_.shape[0])])
    avg = np.array([np.mean(data[kmeans.labels_ == i], axis=0) for i in range(kmeans.cluster_centers_.shape[0])])

    # plot the cluster means with the standard deviation as ribbons
    for i in range(kmeans.cluster_centers_.shape[0]):
        center = avg[i,:]
        std_dev = stds[i,:]
        time = np.arange(center.shape[0])  # Assuming time is a sequence of integers

        plt.plot(time, center, color=colors[i % len(colors)], label='cluster {}'.format(i+1),
                 marker=markers[i % len(markers)], linestyle='-', markersize=4)
        plt.fill_between(time, center - std_dev, center + std_dev, color=colors[i % len(colors)], alpha=0.2)
    plt.ylabel(ylabel if ylabel else f'{column_name} [scaled]')
    plt.xlabel('Time [minutes]')
    plt.legend()
    plt.show()

plot_cluster_means_with_std_dev(trajs, 'radius_eq(ice)[m]', 1e6, 'Radius of ice particles [$\mu$m]')
plot_cluster_means_with_std_dev(trajs, 'density(droplet/ice)[kg/m3]', ylabel='Droplet density s[kg/m3]')