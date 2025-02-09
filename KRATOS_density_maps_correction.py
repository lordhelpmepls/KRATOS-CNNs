#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 7 16:01:01 2025

@author: marlinmahmud
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from KRATOS_module import read_KRATOS
import glob
import os
from astropy.cosmology import FlatLambdaCDM

sim = 'K9'
path = sim+'/'

a_snapshots = np.sort(glob.glob(path+'PMcrs0a*'))
a_snapshots = [i[-10:-4] for i in a_snapshots]
print ('Number of snapshots: ', len(a_snapshots), '/ 61(', np.round(len(a_snapshots)/61*100), '%)')
print ('Snapshots: ', a_snapshots)

output_dir = "density_maps_K9"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def z_func(a):
    return 1/a - 1

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
index_obs = 50  # t=0 Gyr (corresponds to 'a=0.8500')

z_snapshots = z_func(np.array(a_snapshots, dtype=np.float64))
t_snapshots = cosmo.age(z_snapshots).value - cosmo.age(z_snapshots)[0].value
t_snapshots = t_snapshots - t_snapshots[index_obs]

# Alignment function, aligns the LMC disk so its angular momentum points along the z-axis.
def align_face_on(positions, velocities, masses):
    
    L = np.sum(masses[:, None] * np.cross(positions, velocities), axis=0)
    L_hat = L / np.linalg.norm(L)  # Normalize angular momentum

    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(L_hat, z_axis)
    rotation_angle = np.arccos(np.dot(L_hat, z_axis))

    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

        rotated_positions = np.dot(positions, R.T)
        rotated_velocities = np.dot(velocities, R.T)
    else:
        rotated_positions, rotated_velocities = positions, velocities

    return rotated_positions, rotated_velocities

# Apply alignment before density daps 
for a in a_snapshots:
    print(f"Processing snapshot a={a}...")
    
    df = read_KRATOS(sim, a, verbose=False)
    print(f"Dataframe created for a={a}")
    
    if 'galaxy' in df.columns:
        df_LMC = df[df['galaxy'] == 'LMC']  
    else:
        # Spatial filtering
        LMC_center = np.array([0.2, -0.09, -0.25])  # From median positions
        max_LMC_distance = 15  
        distances = np.sqrt((df['x'] - LMC_center[0])**2 +
                            (df['y'] - LMC_center[1])**2 +
                            (df['z'] - LMC_center[2])**2)
        df_LMC = df[distances < max_LMC_distance]

    positions_LMC = np.column_stack((df_LMC['x'], df_LMC['y'], df_LMC['z']))
    velocities_LMC = np.column_stack((df_LMC['vx'], df_LMC['vy'], df_LMC['vz']))
    masses_LMC = df_LMC['mass'].values

    # apply rotation 
    aligned_positions, _ = align_face_on(positions_LMC, velocities_LMC, masses_LMC)

    # Generate density maps using *aligned* positions
    def generate_density_map(positions, masses, grid_size=1024, bounds=[-17, 17]):
        x = positions[:, 0]
        y = positions[:, 1]
        density, _, _ = np.histogram2d(x, y, bins=grid_size, range=[bounds, bounds], weights=masses)
        return density.T  

    def plot_density_map(density, title, bounds=[-17, 17], output_file=None):
        plt.figure(figsize=(12, 12), dpi=600)  

        vmin = np.percentile(density, 1)
        vmax = np.percentile(density, 99)

        norm = Normalize(vmin=vmin, vmax=vmax)  

        plt.imshow(density, origin='lower', extent=[*bounds, *bounds], cmap='magma', norm=norm)
        plt.colorbar(label='Density')
        plt.title(title, fontsize=20, weight='bold')
        plt.xlabel('kpc', fontsize=16)
        plt.ylabel('kpc', fontsize=16)

        if output_file:
            plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)  
        plt.close()  

    # Compute only the face-on density map 
    face_on_density_LMC = generate_density_map(aligned_positions[:, :2], masses_LMC, grid_size=1024)  

    output_face_on = os.path.join(output_dir, f"face_on_LMC_{a}.png")
    
    plot_density_map(face_on_density_LMC, title=f"Face-On Density Map (LMC Only, a={a})", output_file=output_face_on)
    
    print(f"Saved face-on density map for a={a}")
    
    
    