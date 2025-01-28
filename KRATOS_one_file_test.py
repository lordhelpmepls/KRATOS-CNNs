#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:09:20 2025

@author: marlinmahmud
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from KRATOS_module import read_KRATOS

sim = 'K3'
a = '0.8500'

df = read_KRATOS(sim, a, verbose=True)

def generate_density_map(positions, masses, grid_size=1024, bounds=[-50, 50]):
    x = positions[:, 0]
    y = positions[:, 1]
    density, _, _ = np.histogram2d(x, y, bins=grid_size, range=[bounds, bounds], weights=masses)
    return density.T  

def plot_density_map(density, title, bounds=[-50, 50], output_file=None):
    plt.figure(figsize=(12, 12), dpi=600)  
    
    norm = PowerNorm(gamma=0.5)  # Alternative: plt.Normalize() or LogNorm()
    
    plt.imshow(np.arcsinh(density), origin='lower', extent=[*bounds, *bounds], cmap='magma', norm=norm)
    plt.colorbar(label='Asinh Density')
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel('kpc', fontsize=16)
    plt.ylabel('kpc', fontsize=16)
    
    if output_file:
        plt.savefig(output_file, format='tiff', dpi=600, bbox_inches='tight', pad_inches=0.05)  
    
    plt.show()

positions = np.column_stack((df['x'], df['y'], df['z']))
masses = df['mass'].values

face_on_density = generate_density_map(positions[:, :2], masses, grid_size=2048)  
edge_on_density = generate_density_map(positions[:, [0, 2]], masses, grid_size=2048)

plot_density_map(face_on_density, title="Face-On Density Map (a=0.8500)", output_file="face_on_high_res.tiff")
plot_density_map(edge_on_density, title="Edge-On Density Map (a=0.8500)", output_file="edge_on_high_res.tiff")




