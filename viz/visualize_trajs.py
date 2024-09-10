# visualize trajectories
#%% Import libraries
import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

#%% Load data
trajs = pd.read_csv('./trajs_5100_7200_Ns10000_clusters_theta.csv')

#%% Create a pyvista plotter
p = pv.Plotter()
p.open_gif("/Users/psturm/Desktop/LEAP/ice-mp-su24/viz/trajs.gif")

# Extract relevant columns
x = trajs['x[m]'].values
y = trajs['y[m]'].values
z = trajs['z[m]'].values
radius = trajs['radius_eq(ice)[m]'].values
time = trajs['time'].values
clusters = trajs['cluster'].values - 1
theta = trajs['theta'].values

# # Define a custom colormap
unique_clusters = np.unique(clusters)
colors = plt.cm.plasma_r(np.linspace(0, 1, len(unique_clusters) + 1))
colors = colors[1:]  # get rid of that yucky yellow
# Create a new colormap that starts with the first color up to -pi/2
first_color = colors[0]
n_colors = len(colors)
extended_colors = np.vstack(([first_color, first_color], colors))
# Define new color stops
extended_stops = np.linspace(0, 1, n_colors + 2)

# Create the custom colormap
# cmap = LinearSegmentedColormap.from_list("custom_plasma_r", list(zip(extended_stops, extended_colors)))
palette = sns.color_palette("light:b_r", len(unique_clusters))
# make a cmap from the palette
cmap = LinearSegmentedColormap.from_list("custom", palette)
# cmap = 

# Create a sphere source for glyphs
sphere = pv.Sphere(radius=1.0)

unique_times = np.unique(time)
for t in unique_times:
    mask = time == t
    points = np.column_stack((x[mask], y[mask], z[mask]))
    point_cloud = pv.PolyData(points)
    
    # Add cluster information as point data
    point_cloud["cluster"] = clusters[mask]
    point_cloud["theta"] = theta[mask]
    point_cloud["radius"] = radius[mask]
    # point_colors = np.array([cluster_color_map[cluster] for cluster in clusters[mask]])

    # Create glyphs to vary point sizes
    # glyph = point_cloud.glyph(scale="radius", 
    #                           geom=sphere,
    #                           factor=2e6)
    
    p.add_mesh(point_cloud,scalars="theta",
            #    rgb = True,
               render_points_as_spheres=True,
               point_size = 2,
               cmap = cmap,
               show_scalar_bar=False)
    p.view_isometric()
    p.camera.elevation -= 60.0  # Adjust the value as needed
    # p.set_background('skyblue')
    # p.set_background([135/255, 206/255, 250/255])  # RGB for light sky blue
    # p.set_background("daf7fe")  # Hex code for light sky blue
    p.set_background("black")  # nighttime
    p.write_frame()

    

    
# Set the camera to a nice orientation
# p.view_xz()

# Show the plot (interactive by default)
p.show()


print("Saving gif...")
p.close()
print("Saved gif!")

# %%
