# trajectory clusters
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score

import imageio as io

import os
# # for loading the trajectories
workpath = '/glade/work/psturm/ice-mp-su24/'
os.chdir(workpath)
from sd2bin import Bin_Superdroplets

# seed for reproducibility
np.random.seed(42)

# set Ns
# Ns = 10000
Ns = 100000

# load data
rh_trajectories = pd.read_csv('saved_trajectory_data/rh_trajectories_Ns' + str(Ns) + '.csv')
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


kmeans = KMeans(init="k-means++", n_clusters=4, n_init=4)
kmeans.fit(supersaturation)

# plot the cluster centers as a timeseries
plt.figure()

# color code the clusters with a scientific colormap
colors = plt.cm.plasma_r(np.linspace(0, 1, kmeans.cluster_centers_.shape[0]+1))
colors = colors[1:] # get rid of that yucky yellow
color_order = [x - 1 for x in [1, 4, 2, 3]] # reorder for how kmeans find the clusters
colors = colors[color_order[0:len(colors)]]
# make a new colormap with the colors
cmap = sns.color_palette(colors)
# colors = sns.color_palette("light:b_r", kmeans.cluster_centers_.shape[0])
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
plt.savefig('figs/supersaturation_trajectories.png', dpi=300)
plt.show()


# load trajs and add the cluster labels to the dataframe
trajs = pd.read_csv('saved_trajectory_data/trajs_5100_7200_Ns' + str(Ns) + '.csv')
# add the cluster labels to the dataframe
rh_trajectories['cluster'] = kmeans.labels_+1

# left join the cluster labels to the trajs dataframe
trajs = trajs.merge(rh_trajectories['cluster'], left_on='rk_deact', right_index=True)
# save the trajs dataframe with the cluster labels
trajs.to_csv('saved_trajectory_data/trajs_5100_7200_Ns' + str(Ns) + '_clusters.csv')

#%% 
def plot_cluster_means_with_std_dev(trajs, column_name, scale_factor=1, ylabel=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
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

        ax.plot(time, center, color=colors[i % len(colors)], label='cluster {}'.format(i+1),
                marker=markers[i % len(markers)], linestyle='-', markersize=4)
        ax.fill_between(time, center - std_dev, center + std_dev, color=colors[i % len(colors)], alpha=0.2)
    
    ax.set_ylabel(ylabel if ylabel else f'{column_name} [scaled]')
    # ax.set_xlabel('Time [minutes]')
    # ax.legend()
    
    return ax

fig, axs = plt.subplots(2, 2, figsize=(8, 7), sharex=True)
plot_cluster_means_with_std_dev(trajs, 'RH_ice', 100, ylabel='Relative Humidity w.r.t. ice [%]', ax=axs[0, 0])
plot_cluster_means_with_std_dev(trajs, 'radius_eq(ice)[m]', 1e6, 'Radius of ice particles [$\mu$m]', ax=axs[1, 0])
plot_cluster_means_with_std_dev(trajs, 'z[m]', ylabel='Altitude [m]', ax=axs[0,1])
plot_cluster_means_with_std_dev(trajs, 'density(droplet/ice)[kg/m3]', ylabel='Droplet density [kg/m3]', ax=axs[1,1])

axs[1,0].set_xlabel('Time [minutes]')
axs[1,1].set_xlabel('Time [minutes]')
axs[0,0].legend()
# plot_cluster_means_with_std_dev(trajs, 'rhod [kg/m3]', ylabel='deposition density [kg/m3]', ax=axs[1, 1])
plt.tight_layout()
plt.savefig('figs/cluster_means_with_std_dev.png', dpi=300)
plt.show()


#%% KDE plot of the radius distribution for each cluster at each time step

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
## Initialize a list to store the data for each time step
accumulated_data = []
# Loop through each time step and create a plot for it
for i, time_step in enumerate(sorted(unique_times)):
    fig, axs = plt.subplots(len(unique_clusters), 1, 
                            figsize=(10, 2.5 * len(unique_clusters)),
                            sharex=True)
    if len(unique_clusters) == 1:
        axs = [axs]
    
    # Add the current time step's data to the accumulated data
    accumulated_data.append((time_step, i))
    
    for ax, cluster, title_color in zip(axs, unique_clusters, colors):
        cluster_data = trajs[trajs['cluster'] == cluster]
        
        # Plot all accumulated data
        for past_time_step, past_i in accumulated_data:
            time_step_data = cluster_data[cluster_data['time'] == past_time_step]
            
            sns.kdeplot(data=time_step_data, x="radius[microns]", 
                        weights='multiplicity[-]', clip_on=False,
                        color=palette[past_i], label=f'Time {past_time_step}', 
                        alpha=alphas[past_i], bw_adjust=.5, fill=True, ax=ax,
                        clip=(np.min(trajs["radius[microns]"]), None), cut=0)
        
        ax.set_ylabel(f'Cluster {cluster}', fontsize=20)
        ax.set_yticks([])
        ax.spines['left'].set_color(title_color)
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_color(title_color)
        ax.spines['bottom'].set_linewidth(3)
        ax.tick_params(axis='x', which='major', labelsize=15)
        ax.tick_params(axis='x', which='minor', labelsize=15)
        ax.set_xlabel('Ice crystal radius [$\mu$m]', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(f'kde_frames/frame_{i:03d}.png', dpi = 300)
    plt.close(fig)

# Create a GIF from the saved frames
with io.get_writer('kde_clusters.gif', mode='I', duration=0.5, loop = 0) as writer:
# with io.get_writer('kde_clusters.gif', mode='I', duration=0.5) as writer:

    for i in range(len(unique_times)):
        filename = f'kde_frames/frame_{i:03d}.png'
        image = io.imread(filename)
        writer.append_data(image)


# sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# # Filter for Cluster 3
# cluster_3_data = trajs[trajs['cluster'] == 2]

# # Create a new column 'g' for FacetGrid, converting 'time' to a string for hue mapping
# cluster_3_data['g'] = cluster_3_data['time'].astype(str)

# # Initialize the FacetGrid object with a smaller height for each subplot
# pal = "crest_r"
# g = sns.FacetGrid(cluster_3_data, row="g", hue="g", aspect=15, height=.3, palette=pal)  # Reduced height

# # Draw the densities
# g.map(sns.kdeplot, "radius[microns]",
#       bw_adjust=.5, clip_on=False,
#       fill=True, alpha=1, linewidth=1.5)
# g.map(sns.kdeplot, "radius[microns]", clip_on=False, color="w", lw=2, bw_adjust=.5)

# # Passing color=None to refline() uses the hue mapping
# g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# # Define and use a simple function to label the plot in axes coordinates
# def label(x, color, label):
#     ax = plt.gca()
#     ax.text(0, .2, label, color=color,
#             ha="left", va="center", transform=ax.transAxes)

# g.map(label, "radius[microns]")

# # Set the subplots to overlap more by adjusting hspace
# g.figure.subplots_adjust(hspace=-.8)  # Increased overlap

# # Remove axes details that don't play well with overlap
# g.set_titles("")
# g.set(yticks=[], ylabel="")
# g.despine(bottom=True, left=True)

# plt.show()

#%% Do 2D PCA 

# do PCA on two components
pca = PCA(n_components=2)
pca.fit(supersaturation)
supersaturation_pca = pca.transform(supersaturation)

# plot the first two principal components
plt.figure()
# get the percentage of points in each cluster
cluster_sizes = np.bincount(kmeans.labels_)/len(kmeans.labels_)*100
for i in range(kmeans.cluster_centers_.shape[0]):
    cluster_label = f'cluster {i+1} ({cluster_sizes[i]:.2f}%)'
    plt.scatter(supersaturation_pca[kmeans.labels_ == i, 0], 
                supersaturation_pca[kmeans.labels_ == i, 1], 
                label=cluster_label,
                color=colors[i % len(colors)],
                s=3, alpha=0.4)
    # plt.scatter(pc1_spiral_fit, pc2_spiral_fit, color='black', 
    #         s=0.1, alpha=0.4)
    # plot all of cluster 1 with radius < 5 microns
    if i == 0:
        radius_trajectories = trajs.pivot(index='rk_deact', columns='time', values='radius_eq(ice)[m]')
        plt.scatter(supersaturation_pca[(kmeans.labels_ == i) & (radius_trajectories[7200]<5e-6), 0], 
                    supersaturation_pca[(kmeans.labels_ == i) & (radius_trajectories[7200]<5e-6), 1], 
                    label='cluster 1, final radius < 5 microns',
                    color='black',
                    s=3, alpha=0.4)
plt.legend()
plt.xlabel('PC1, explained variance ratio: {:.1f}'.format(pca.explained_variance_ratio_[0]*100) + '%')
plt.ylabel('PC2, explained variance ratio: {:.1f}'.format(pca.explained_variance_ratio_[1]*100) + '%')

# plot the weights of the first two principal components
plt.figure()
plt.plot(np.arange(pca.components_.shape[1]), 
        pca.components_[0,:], 
        label='1st principal component')
plt.plot(np.arange(pca.components_.shape[1]), 
        pca.components_[1,:], 
        label='2nd principal component',
        alpha=0.5)
plt.xlabel('Time [minutes]')
plt.ylabel('Principal component weight')

# dot the principal component weights with the height z[m]
# organize this by cluster label
plt.figure()
height = trajs.pivot(index='rk_deact', columns='time', values='z[m]')
diff_height = height.diff(axis=1)
# drop 5100 because it's the first time step
diff_height = diff_height.drop(columns=5100)
proj1 = np.dot(diff_height, pca.components_[0,1:])
proj2 = np.dot(diff_height, pca.components_[1,1:])
for i in range(kmeans.cluster_centers_.shape[0]):
    plt.scatter(proj1[kmeans.labels_ == i],
                proj2[kmeans.labels_ == i],
                label='cluster {}'.format(i+1),
                color=colors[i % len(colors)],
                s = 1, alpha = 0.2)
plt.xlabel('1st principal component dot height')
plt.ylabel('2nd principal component dot height')

# fit a spiral to the 2D PCA data
# make the spiral a function of t and fit parameters a, b, c, d
# and return 2D coordinates x, y
# curve_fit only allows a single dependent variable, so we need to


#%% fit a spiral to the 2D PCA data
from scipy.optimize import curve_fit

# subsample the data to make the spiral fit faster and represent each cluster equally
min_samples_in_cluster = np.min(np.bincount(kmeans.labels_))
max_samples = min_samples_in_cluster
# max_samples = 100000

# Initialize lists to hold the subsampled data and labels
subsampled_data_list = []
subsampled_labels_list = []

# Iterate through each cluster to subsample
for cluster_id in range(kmeans.n_clusters):
    # Find indices of samples in the current cluster
    cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
    # If the cluster size is larger than max_samples, truncate it
    if len(cluster_indices) > max_samples:
        selected_indices = cluster_indices[:max_samples]
    else:
        selected_indices = cluster_indices   
    # Append the selected samples and their labels to the lists
    subsampled_data_list.append(supersaturation_pca[selected_indices, :])
    subsampled_labels_list.extend([cluster_id] * len(selected_indices))

# Concatenate the lists to form the subsampled dataset and labels array
supersat_PCA_subsampled = np.vstack(subsampled_data_list)
subsampled_labels = np.array(subsampled_labels_list)


# convert pc1 and pc2 to polar coordinates
r = np.sqrt(supersat_PCA_subsampled[:,0]**2 + supersat_PCA_subsampled[:,1]**2)
theta = np.arctan2(supersat_PCA_subsampled[:,1], 
                   supersat_PCA_subsampled[:,0])

# some hand tuning for theta
# if theta<0 AND r>1, add 2pi to theta
# theta[(theta < -2) & (r > 1)] += 2*np.pi
# if theta>0 AND r<0.08, subtract 2pi from theta
theta[(theta > 0) & (r < 0.2)] -= 2*np.pi
# if theta < 0 AND r > 1, add 2pi to theta
theta[(theta < -1) & (r > 0.8)] += 2 * np.pi
# but correct this for the cluster 3 data, so add 2pi to theta
# when cluster 3 and theta < -4
theta[(subsampled_labels == 2) & (theta < -np.pi)] += 2 * np.pi

# logarithmic spiral
# def spiral(theta, a, b, c):
#     a = np.exp(1)
#     r = 1/a * np.exp(b * theta) - 1/a 
#     return r



def spiral(theta, a, b):
    # make this one a golden spiral squared
    phi = (1 + np.sqrt(5)) / 2
    # if wanting to fit a only, set b to phi
    # a = 0.172
    # b = 0
    # b = phi
    # a = 1.3 # for 100k superdroplets
    # c = 0
    r = a * (phi*b) ** ( 2 * (theta) / np.pi)
    return r

# def golden_spiral(theta):
#     # log spiral just for fitting checking
#     phi = (1 + np.sqrt(5)) / 2
#     a = np.pi/17
#     b = np.pi/5
#     c = -0.03
#     # b = np.pi/2
#     r =  a * np.exp(b * theta) + c
#     return r

def golden_spiral(theta):
    phi = (1 + np.sqrt(5)) / 2
    a = 0.171 #
    # a = phi # for 100k superdroplets
    b = phi
    c = 0
    # c = -0.03
    # b = np.pi/2
    r = a * (b*phi) ** ( 2 * theta / np.pi) + c
    # r = 0.1  * phi ** ( 4 * theta / np.pi + 1)
    return r

# def golden_spiral(theta):
#     phi = (1 + np.sqrt(5)) / 2
#     r = 0.3 * phi ** ( 2*  theta / np.pi)
#     return r

# constrain the parameters
bounds = ([0, 0] , [10. , 10.])
# bounds = ([0, 1.0, -.1], [0.4, 4.0, 0])

# fit the spiral to the data
popt, pcov = curve_fit(spiral, theta, r, bounds=bounds)

# generate the spiral
# theta_fit = np.linspace(theta.min(), theta.max(), 1000)
theta_fit = np.linspace(theta.min(), theta.max(), 1000)
r_fit = spiral(theta_fit, *popt)

r_golden = golden_spiral(theta_fit)

pc1_golden = r_golden * np.cos(theta_fit)
pc2_golden = r_golden * np.sin(theta_fit)

# convert back to cartesian coordinates
pc1_spiral_fit = r_fit * np.cos(theta_fit)
pc2_spiral_fit = r_fit * np.sin(theta_fit)


phi = (1 + np.sqrt(5)) / 2
a,b = popt
agold = .171
fit_label = f'$r = {a:.3f} \cdot ({b:.3f} \cdot \phi)^{{2 \\theta / \pi}}$'
gold_label = f'$r = {agold:.3f} \cdot \phi ^{{4 \\theta / \pi}}$'
# gold_label = f'$r = 0.1 \cdot \phi ^{{4 \\theta / \pi + 1}}$'

# plot scatter in polar coordinates
plt.figure()
for i in range(kmeans.cluster_centers_.shape[0]):
    plt.scatter(theta[subsampled_labels == i], r[subsampled_labels == i], 
                label='cluster {}'.format(i+1),
                color=colors[i % len(colors)],
                s = 2, alpha = 0.4)
# plot spiral fits at different widths so they are visible and pretty
plt.plot(theta_fit, r_fit, color='black', 
         label=fit_label, linewidth = 3.5)
plt.plot(theta_fit, r_golden, color='gold', 
         label=gold_label, linewidth = 1.0)
plt.xlabel('$\\theta$')
plt.ylabel('Spiral radius')
plt.legend()
plt.savefig('figs/spiral_fit_polar.png', dpi=300)

# plot the golden spiral on top of the data
plt.figure()
for i in range(kmeans.cluster_centers_.shape[0]):
    plt.scatter(supersat_PCA_subsampled[subsampled_labels == i, 0], 
                supersat_PCA_subsampled[subsampled_labels == i, 1], 
                label='cluster {}'.format(i+1),
                color=colors[i % len(colors)],
                s = 2, alpha = 0.4)
# plot spiral fits at different widths so they are visible and pretty
plt.plot(pc1_spiral_fit, pc2_spiral_fit, 
         color='black', label = fit_label,
         linewidth = 3.5) 
plt.plot(pc1_golden, pc2_golden, 
         color='gold', label=gold_label,
         linewidth = 1)


# set xlims and ylims to be the limits of the data
plt.xlim([supersat_PCA_subsampled[:,0].min(), supersat_PCA_subsampled[:,0].max()])
plt.ylim([supersat_PCA_subsampled[:,1].min(), supersat_PCA_subsampled[:,1].max()])

plt.xlabel('PC1, explained variance ratio: {:.1f}'.format(pca.explained_variance_ratio_[0]*100) + '%')
plt.ylabel('PC2, explained variance ratio: {:.1f}'.format(pca.explained_variance_ratio_[1]*100) + '%')
# make a title with the golden spiral equation
phi = (1 + np.sqrt(5)) / 2
a,b = popt
# a, b, c = popt
# plt.title(f'$r = {a:.3f} \cdot ( phi)^{{4 \\theta / \pi}} + {b:.3f}$, where $\phi = (1 + \sqrt{{5}}) / 2$')
# plt.title(f'$r = {a:.3f} \cdot ({b:.3f} \cdot \phi)^{{2 \\theta / \pi}}$, where $\phi = (1 + \sqrt{{5}}) / 2$')
# plt.title(f'$r = {a:.3f} \cdot ({b:.3f} \cdot phi )^{{2 \\theta / \pi}} - {-c:.3f}$')
# plt.title(f'$r = {a:.3f} \cdot e^{{ {b:.3f} \\theta}} - {-c:.3f}$')
plt.legend()
plt.savefig('figs/spiral_fit.png', dpi=300)
# plt.savefig('figs/pca_nofit.png', dpi=300)



#%% Zoom in on cluster 1

# # find subclusters in cluster 1 
cluster1 = supersaturation[kmeans.labels_ == 0,:]

# cluster 1 pca is the supersaturation pca for cluster 1
cluster1_pca = supersaturation_pca[kmeans.labels_ == 0,:]

# plot a star at theta=0.01 and theta=0.02
# convert to polar coordinates
theta1 = np.min(theta)
theta2 = -2.8
theta3 = 0
# theta4 = -2.8
# theta5 = -0.75
r1 = spiral(theta1, *popt)
r2 = spiral(theta2, *popt)
r3 = spiral(theta3, *popt)
# r4 = spiral(theta4, *popt)
# r5 = spiral(theta5, *popt)
x1,y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
x2,y2 = r2 * np.cos(theta2), r2 * np.sin(theta2)
x3,y3 = r3 * np.cos(theta3), r3 * np.sin(theta3)
x4,y4 = r2 * np.cos(theta2)-0.1, r2 * np.sin(theta2)
# x4,y4 = r4 * np.cos(theta4), r4 * np.sin(theta4)
# x5,y5 = r5 * np.cos(theta5), r5 * np.sin(theta5)

# map x1,y1 and x2,y2 back to the supersaturation space
# decode from pca space to supersaturation space
sat1 = pca.inverse_transform([x1, y1])
sat2 = pca.inverse_transform([x2, y2]) 
sat3 = pca.inverse_transform([x3, y3])
sat4 = pca.inverse_transform([x4, y4])

# Create subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5, 8))

for i in range(kmeans.cluster_centers_.shape[0]):
    cluster_label = f'cluster {i+1} ({cluster_sizes[i]:.2f}%)'
    ax1.scatter(supersaturation_pca[kmeans.labels_ == i, 0], 
                supersaturation_pca[kmeans.labels_ == i, 1], 
                color=colors[i % len(colors)],
                s=3, alpha=0.4)
    if i == 0:
        radius_trajectories = trajs.pivot(index='rk_deact', columns='time', values='radius_eq(ice)[m]')
        black_points = ax1.scatter(supersaturation_pca[(kmeans.labels_ == i) & (radius_trajectories[7200]<5e-6), 0], 
                                   supersaturation_pca[(kmeans.labels_ == i) & (radius_trajectories[7200]<5e-6), 1], 
                                   label='cluster 1, final r < 5 $\mu$m',
                                   color='black',
                                   s=3, alpha=0.4)
ax1.plot(pc1_spiral_fit, pc2_spiral_fit, color='black', 
         label=fit_label, linewidth=3.5)
ax1.plot(pc1_golden, pc2_golden, color='gold',
         label=gold_label, linewidth=1)
ax1.set_xlim([cluster1_pca[:,0].min(), cluster1_pca[:,0].max()])
ax1.set_ylim([cluster1_pca[:,1].min(), cluster1_pca[:,1].max()])
ax1.set_xlabel('PC1, explained variance ratio: {:.1f}'.format(pca.explained_variance_ratio_[0]*100) + '%')
ax1.set_ylabel('PC2, explained variance ratio: {:.1f}'.format(pca.explained_variance_ratio_[1]*100) + '%')

# Plot the stars
theta1_marker = ax1.scatter(x1, y1, color='red', marker='.', s=350, 
                       edgecolor='white', linewidths=0.5, 
                       label='$\\theta$=' + f'{theta1:.2f}')
theta2_marker = ax1.scatter(x2, y2, color='green', marker='*', s=350,
                            edgecolor='white', linewidths=0.5, 
                            label='$\\theta$=' + f'{theta2:.2f}')
theta3_marker = ax1.scatter(x3, y3, color='purple', marker='^', s=300,
                            edgecolor='white', linewidths=0.5, 
                            label='$\\theta$=' + f'{theta3:.2f}')
theta4_marker = ax1.scatter(x4, y4, color='blue', marker='s', s=200,
                            edgecolor='white', linewidths=0.5,
                            label='$\\theta$=' + f'{theta3:.2f}, x offset of -0.1')

# Make the legend only for the stars and the black points
ax1.legend(handles=[black_points, theta1_marker, 
                    theta2_marker, theta3_marker, theta4_marker],)

# Plot the second subplot
ax2.scatter(np.arange(len(sat1)), sat1, color='red', marker='.', 
            label='$\\theta$=' + f'{theta1:.2f}')
ax2.scatter(np.arange(len(sat2)), sat2, color='green', marker='*',
            label = '$\\theta$=' + f'{theta2:.2f}')
ax2.scatter(np.arange(len(sat3)), sat3, color='purple', marker='^',
            label='$\\theta$=' + f'{theta3:.2f}')
ax2.scatter(np.arange(len(sat4)), sat4, color='blue', marker='s',
            label='$\\theta$=' + f'{theta3:.2f}, x offset of -0.08')
ax2.set_xlabel('Time [minutes]')
ax2.set_ylabel('Supersaturation')
# Get current y-axis limits
ymin, ymax = ax2.get_ylim()
# color ax2 background yellow below y = 0 and blue above
ax2.axhspan(ymin, 0, color='yellow', alpha=0.15)
ax2.axhspan(0, ymax, color='lightblue', alpha=0.15)

# ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()







#%% Assign theta to each superdroplet (their IDs are rk_deact)

r = np.sqrt(supersaturation_pca[:,0]**2 + supersaturation_pca[:,1]**2)
theta = np.arctan2(supersaturation_pca[:,1], 
                   supersaturation_pca[:,0])

# some hand tuning for theta
# if theta<0 AND r>1, add 2pi to theta
# theta[(theta < -2) & (r > 1)] += 2*np.pi
# if theta>0 AND r<0.08, subtract 2pi from theta
theta[(theta > 0) & (r < 0.2)] -= 2*np.pi
# if theta < 0 AND r > 1, add 2pi to theta
theta[(theta < -1) & (r > 0.8)] += 2 * np.pi
# but correct this for the cluster 3 data, so add 2pi to theta
# when cluster 3 and theta < -4
theta[(kmeans.labels_ == 2) & (theta < -np.pi)] += 2 * np.pi


# assign theta to each superdroplet in rh_trajectories
theta_dict = dict(zip(rh_trajectories.index, theta))

# make a new dataframe with the theta values
# Drop all columns in trajs that start with 'theta'
theta_columns = trajs.filter(like='theta').columns
trajs.drop(columns=theta_columns, inplace=True)
theta_df = pd.DataFrame.from_dict(theta_dict, orient='index', columns=['theta'])

# left join the theta values to the trajs dataframe
trajs = trajs.merge(theta_df, left_on='rk_deact', right_index=True, suffixes=('', '_theta'))

# save the trajs dataframe with the theta values
trajs.to_csv('saved_trajectory_data/trajs_5100_7200_Ns' + str(Ns) + '_cluster_theta.csv')


# %% predict theta from instantaneous conditions
# this includes environmental conditions (T, p, rh)
# as well as microphysical characteristics of the superdroplet (r, rhod)

# let's use gradient boosting to predict theta from the instantaneous conditions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = trajs[['T [K]', 'prs', 'rh', 'radius_eq(ice)[m]', 'rhod [kg/m3]']]
y = trajs['theta']

# Shuffle the indices
indices = np.arange(X.shape[0])

# Split the indices
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create the training and test sets using the indices
X_train, X_test = X.loc[train_indices], X.loc[test_indices]
y_train, y_test = y.loc[train_indices], y.loc[test_indices]
# keep track of rk_deact for each set too
train_rk_deact = trajs.iloc[train_indices]['rk_deact']
test_rk_deact = trajs.iloc[test_indices]['rk_deact']

#%% How well can we unravel the spiral with the PCA components?

# get the spiral fit in the supersaturation space
# convert back to cartesian coordinates

predicted_radius = spiral(theta, *popt)

# show r2 fit of the radius given theta
r2 = r2_score(r, predicted_radius)

# plot the predicted radius vs the actual radius
plt.figure()
for i in range(kmeans.cluster_centers_.shape[0]):
    plt.scatter(r[kmeans.labels_ == i], 
                predicted_radius[kmeans.labels_ == i], 
                label='cluster {}'.format(i+1),
                color=colors[i % len(colors)],
                s=3, alpha=0.4)
# get ylims to plot the 1:1 line
ylim = plt.gca().get_ylim()
plt.plot(ylim, ylim, color='red', linestyle='--')
plt.xlabel('actual radius')
plt.ylabel('predicted radius')
# make loglog
# plt.xscale('log')
# plt.yscale('log')
plt.title(f'Predicted radius from theta\nR2: {r2:.2f}')
plt.show()

# coordinate transformation to PC space
predicted_pc1 = predicted_radius * np.cos(theta)
predicted_pc2 = predicted_radius * np.sin(theta)

# unravel pc1 and pc2 to supersaturation space
predicted_supersaturation = pca.inverse_transform(np.vstack([predicted_pc1, predicted_pc2]).T)
decompressed_supersaturation = pca.inverse_transform(supersaturation_pca)

# make the kmeans labels the same shape as the predicted supersaturation
# this means we need to repeat the labels for each time step
kmeans_labels_time = np.array([kmeans.labels_]*(supersaturation.shape[1])).T

# show r2 fit of the supersaturation given theta
r2 = r2_score(supersaturation, predicted_supersaturation)
# plot the predicted supersaturation vs the actual supersaturation
flat_supersaturation = supersaturation.flatten()
flat_predicted_supersaturation = predicted_supersaturation.flatten()
flat_kmeans_labels_time = kmeans_labels_time.flatten()
plt.figure()
for i in range(kmeans.cluster_centers_.shape[0]):
    plt.scatter(flat_supersaturation[flat_kmeans_labels_time == i], 
                flat_predicted_supersaturation[flat_kmeans_labels_time == i], 
                label='cluster {}'.format(i+1),
                color=colors[i % len(colors)],
                s=3, alpha=0.4)
# get ylims to plot the 1:1 line
ylim = plt.gca().get_ylim()
plt.plot(ylim, ylim, color='red', linestyle='--')

plt.xlabel('actual supersaturation')
plt.ylabel('predicted supersaturation')
plt.title(f'Predicted supersaturation from theta\nR2: {r2:.2f}')

# plot a timeseries of r2 values, one for each time step
r2_spiral = np.zeros(supersaturation.shape[1])
r2_pca = np.zeros(supersaturation.shape[1])
for i in range(supersaturation.shape[1]):
    r2_spiral[i] = r2_score(supersaturation[:,i], predicted_supersaturation[:,i])
    r2_pca[i] = r2_score(supersaturation[:,i], decompressed_supersaturation[:,i])
plt.figure()
plt.plot(r2_spiral, label='Generated by golden spiral')
plt.plot(r2_pca, label='Reconstructed from PCA ')
plt.xlabel('Timestep [minutes]')
plt.ylabel('Supersaturation R2 score')
# ylim should be 0, 1
plt.ylim([0, 1])
plt.legend()
plt.show()


#%% fit the gradient boosting regressor
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=10, 
                                learning_rate=0.1, loss='squared_error',
                                verbose=1)
gbr.fit(X_train, y_train)

# predict theta
y_pred = gbr.predict(X_test)

# calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# calculate the r2 score
r2 = gbr.score(X_test, y_test)

#%% plot the predicted vs actual theta, color by cluster
# Convert y_test and y_pred to pandas Series
y_test_series = pd.Series(y_test, index=X_test.index)
y_pred_series = pd.Series(y_pred, index=X_test.index)

plt.figure()
for i in range(kmeans.cluster_centers_.shape[0]):
    print(i)
    cluster_indices = test_indices[trajs.loc[test_indices, 'cluster'] == i+1]
    plt.scatter(y_test_series.loc[cluster_indices], 
                y_pred_series.loc[cluster_indices], 
                label='cluster {}'.format(i+1),
                color=colors[i % len(colors)],
                s=3, alpha=0.4)
plt.plot([y_test_series.min(), y_test_series.max()], [y_test_series.min(), y_test_series.max()], 
         color='red', linestyle='--')
plt.xlabel('actual theta')
plt.ylabel('predicted theta')
plt.title(f'Gradient boosting regression\nMSE: {mse:.2f}, R2: {r2:.2f}')
plt.legend()
plt.show()



# %%
