#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:09:20 2025

@author: marlinmahmud
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from KRATOS_module import read_KRATOS
import glob
from astropy.cosmology import FlatLambdaCDM

sim = 'K3'
a = '0.8500'

path = sim+'/'

a_snapshots = np.sort(glob.glob(path+'PMcrs0a*'))
a_snapshots = [i[-10:-4] for i in a_snapshots]
print ('Number of snapshots: ',len(a_snapshots),'/ 61(', np.round(len(a_snapshots)/61*100), '%)')
print (a_snapshots)

def z_func(a):
    z = 1/a - 1
    return z

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

index_obs = 50  # t=0 Gyr (corresponds to 'a=0.8500')

z_snapshots = z_func(np.array(a_snapshots, dtype=np.float64))
t_snapshots = cosmo.age(z_snapshots).value-cosmo.age(z_snapshots)[0].value


df = read_KRATOS(sim,a,verbose=False)


if 'galaxy' in df.columns:
    df_LMC = df[df['galaxy'] == 'LMC']  
else:
    # Spatial filtering
    LMC_center = np.array([0.2, -0.09, -0.25])  # Median positions
    max_LMC_distance = 15

    # Distance from LMC center
    distances = np.sqrt((df['x'] - LMC_center[0])**2 +
                        (df['y'] - LMC_center[1])**2 +
                        (df['z'] - LMC_center[2])**2)

def generate_density_map(positions, masses, grid_size=1024):
    x = positions[:, 0]
    y = positions[:, 1]
    density, _, _ = np.histogram2d(x, y, bins=grid_size, weights=masses)
    return density.T  

def plot_density_map(density, title, output_file=None):
    plt.figure(figsize=(12, 12), dpi=600)  
    
    vmin = np.percentile(density, 1)
    vmax = np.percentile(density, 99)

    norm = Normalize(vmin=vmin, vmax=vmax)    

    plt.imshow(density, origin='lower', cmap='magma', norm=norm)
    plt.colorbar(label='Density')
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel('kpc', fontsize=16)
    plt.ylabel('kpc', fontsize=16)

    if output_file:
        plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)  
    
    plt.show()

positions_LMC = np.column_stack((df_LMC['x'], df_LMC['y'], df_LMC['z']))
masses_LMC = df_LMC['mass'].values

# Generate density maps
face_on_density_LMC = generate_density_map(positions_LMC[:, :2], masses_LMC, grid_size=1024)  
edge_on_density_LMC = generate_density_map(positions_LMC[:, [0, 2]], masses_LMC, grid_size=1024)

# Plot LMC-only density maps
plot_density_map(face_on_density_LMC, title="Face-On Density Map (LMC Only, a=0.9000)", output_file="face_on_LMC_0.9000.png")
plot_density_map(edge_on_density_LMC, title="Edge-On Density Map (LMC Only, a=0.9000)", output_file="edge_on_LMC_0.9000.png")

