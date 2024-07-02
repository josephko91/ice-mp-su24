[Work in Progress]  
Setting up 3dviz conda env using yml file:    
`conda env create -n [env-name] -f 3dviz_env.yml`

Setting up 3dviz conda env from scratch:  
```
conda env create -n [env-name]
conda activate [env-name]
conda install -c conda-forge pyvista jupyterlab trame-vtk trame-vuetify xarray pyvista-xarray
```
