# trajectory clusters
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
# # for loading the trajectories
workpath = '/glade/work/psturm/ice-mp-su24/'
os.chdir(workpath)
from sd2bin import Bin_Superdroplets

# seed for reproducibility
np.random.seed(42)


# load data
rh_trajectories = pd.read_csv('saved_trajectory_data/rh_trajectories_Ns10000.csv')
rh_trajectories = rh_trajectories.dropna()

# make the index rk_deact
rh_trajectories = rh_trajectories.set_index('rk_deact')

# convert to numpy array
supersaturation = (rh_trajectories.to_numpy()-1)

# histogram of supersaturation
plt.hist(supersaturation.flatten()*100, bins=100)
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
kmeans_PCA = KMeans(init="k-means++", n_clusters=5, n_init=4)
kmeans_PCA.fit(reduced_data)
# Get the explained variance ratio of the principal components
explained_variance_ratio = PCA(n_components=2).fit(supersaturation).explained_variance_ratio_


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans_PCA.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.plasma,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "bx", markersize=2)
# Get the percentage of points in each cluster
cluster_sizes = np.bincount(kmeans_PCA.labels_)/len(kmeans_PCA.labels_)*100
for i in range(kmeans_PCA.cluster_centers_.shape[0]):
    text = plt.text(
        kmeans_PCA.cluster_centers_[i, 0],
        kmeans_PCA.cluster_centers_[i, 1],
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
trajs = pd.read_csv('saved_trajectory_data/trajs_5100_7200_Ns10000.csv')
# add the cluster labels to the dataframe
rh_trajectories['cluster'] = kmeans.labels_+1

# left join the cluster labels to the trajs dataframe
trajs = trajs.merge(rh_trajectories['cluster'], left_on='rk_deact', right_index=True)
# save the trajs dataframe with the cluster labels
trajs.to_csv('saved_trajectory_data/trajs__5100_7200_Ns10000_clusters.csv', index=False)


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
plot_cluster_means_with_std_dev(trajs, 'z[m]', ylabel='Altitude [m]')



# Identify unique time steps and clusters
# set trajs radius[microns] column
trajs['radius[microns]'] = trajs['radius_eq(ice)[m]'] * 1e6
unique_times = trajs['time'].unique()
unique_clusters = np.unique(kmeans.labels_ + 1)
# Create a color palette
num_colors = len(unique_times)
palette = sns.color_palette("crest_r", num_colors)
# palette = sns.color_palette("light:b_r", num_colors)
# Calculate alpha values
alphas = np.linspace(0.9, 0.1, num=len(unique_times))
# Create a figure to hold all subplots
fig, axs = plt.subplots(len(unique_clusters), 1, 
                        figsize=(10, 2.5 * len(unique_clusters)),
                        sharex=True)
# If there's only one cluster, axs may not be an array, ensure it's iterable
if len(unique_clusters) == 1:
    axs = [axs]
# Loop through each cluster and create a subplot for it
for ax, cluster, title_color in zip(axs, unique_clusters, colors):
    cluster_data = trajs[trajs['cluster'] == cluster]
    
    for i, time_step in enumerate(sorted(unique_times)):
        time_step_data = cluster_data[cluster_data['time'] == time_step]
        sns.kdeplot(data=time_step_data, x="radius[microns]", 
                    weights='multiplicity[-]', clip_on=False,
                    color=palette[i], label=f'Time {time_step}', 
                    alpha=alphas[i], bw_adjust=.5, fill=True, ax=ax,
                    clip=(np.min(trajs["radius[microns]"]), None), cut=0)
    
    # Remove the set_title call
    # ax.set_title(f'Cluster {cluster} size distribution',
    #              fontsize=35, fontweight='bold',
    #              color=title_color,
    #              path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    
    # Set the y-axis label to "Cluster <number>"
    ax.set_ylabel(f'Cluster {cluster}', fontsize=20, fontstyle='italic', color=title_color,
                  path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    
    # Other customization remains the same
    ax.set_yticks([])
    # ax.set_ylabel('')
    ax.spines['left'].set_color(title_color)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color(title_color)
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='x', which='minor', labelsize=15)
    ax.set_xlabel('Ice particle radius [$\mu$m]', fontsize=20)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()




sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# Filter for Cluster 3
cluster_3_data = trajs[trajs['cluster'] == 2]

# Create a new column 'g' for FacetGrid, converting 'time' to a string for hue mapping
cluster_3_data['g'] = cluster_3_data['time'].astype(str)

# Initialize the FacetGrid object with a smaller height for each subplot
pal = "crest_r"
g = sns.FacetGrid(cluster_3_data, row="g", hue="g", aspect=15, height=.3, palette=pal)  # Reduced height

# Draw the densities
g.map(sns.kdeplot, "radius[microns]",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "radius[microns]", clip_on=False, color="w", lw=2, bw_adjust=.5)

# Passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "radius[microns]")

# Set the subplots to overlap more by adjusting hspace
g.figure.subplots_adjust(hspace=-.8)  # Increased overlap

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

plt.show()