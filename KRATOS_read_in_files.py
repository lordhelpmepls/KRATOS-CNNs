#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:09:20 2025

@author: marlinmahmud
"""

### TESTED - OK


import numpy as np
import pandas as pd
import glob
from astropy.cosmology import FlatLambdaCDM
from KRATOS_module import read_KRATOS
import matplotlib.pyplot as plt
import os


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
t_snapshots = t_snapshots - t_snapshots[index_obs]
t_snapshots


df = read_KRATOS(sim,a,verbose=True)

df

# Create a 2D density map by projecting 3D particle data
def generate_density_map(positions, masses, grid_size=256, bounds=[-50, 50]):
    x = positions[:, 0]
    y = positions[:, 1]
    density, _, _ = np.histogram2d(x, y, bins=grid_size, range=[bounds, bounds], weights=masses)
    return density.T  # Transpose for correct orientation

# Single density map
def plot_density_map(density, title, bounds=[-50, 50]):
    plt.imshow(np.log10(density + 1), origin='lower', extent=[*bounds, *bounds], cmap='viridis')
    plt.colorbar(label='Log10 Density')
    plt.title(title)
    plt.xlabel('kpc')
    plt.ylabel('kpc')
    plt.show()


output_folder = "density_maps_green_K3/"
os.makedirs(output_folder, exist_ok=True)

for a in a_snapshots:
    print(f"Processing snapshot: {a}")

    df = read_KRATOS(sim, a, verbose=False)

    positions = np.column_stack((df['x'], df['y'], df['z']))
    masses = df['mass'].values

    face_on_density = generate_density_map(positions[:, :2], masses, grid_size=256)
    edge_on_density = generate_density_map(positions[:, [0, 2]], masses, grid_size=256)

    plt.imsave(f"{output_folder}/face_on_a{a}.png", np.log10(face_on_density + 1), cmap='viridis', dpi=300)
    plt.imsave(f"{output_folder}/edge_on_a{a}.png", np.log10(edge_on_density + 1), cmap='viridis', dpi=300)

    print(f"Saved density maps for snapshot: {a}")

