#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 06:34:30 2024

@author: seba

This script models and analyzes neural dynamics in a simulated network using Nengo, focusing on the emergence and representation of oscillatory activity within a neural ensemble. The process begins by defining a neural network where an initial brief input pulse triggers sustained oscillations through feedback connections. The simulation captures the firing patterns of neurons over time, which are then processed to smooth the activity for further analysis.

To understand the underlying structure of these neural activities, principal component analysis (PCA) is employed. PCA reduces the high-dimensional neural data to its most significant components, providing insights into the dominant patterns of neural dynamics. The first three principal components are examined in detail, revealing how the neural population’s activity evolves over time.

Visualization of the results is twofold: first, a raster plot displays the firing activity of individual neurons, highlighting the temporal structure of the network’s oscillations. Second, the principal components are plotted against time and in a 3D space, offering a clear view of how the network’s collective dynamics unfold in reduced dimensions. The use of time as a color gradient in the 3D plot connects the neural activity patterns to their temporal evolution, making it easier to track the progression of neural states.

Overall, this script demonstrates how a combination of simulation, data processing, and dimensionality reduction techniques can be used to explore and visualize the complex dynamics of neural networks, providing a deeper understanding of their emergent behaviors.

"""

import nengo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

np.random.seed(22)

# Define your sampling interval
dt = 0.001  # Example: 1 ms
T  = 3.0    # simulation time, s

# Define the Nengo model
model = nengo.Network()
with model:
    # Create the ensemble for the oscillator
    neurons = nengo.Ensemble(n_neurons=500, dimensions=2, 
        max_rates=nengo.dists.Uniform(2, 100), neuron_type=nengo.LIF())

    # Create an input signal that gives a brief input pulse to start the oscillator
    stim = nengo.Node(lambda t: 1 if t < 0.1 else 0)
    # Connect the input signal to the neural ensemble
    nengo.Connection(stim, neurons[0])

    # Create the feedback connection
    nengo.Connection(neurons, neurons, transform=[[1, 1], [-1, 1]], synapse=0.1)

    # Probe the neural activity
    probe = nengo.Probe(neurons.neurons)

# Run the simulation
with nengo.Simulator(model, dt=dt) as sim:
    sim.run(T)

# Retrieve and process the neural data
neural_data = np.array(sim.data[probe], copy=True)
neural_data[neural_data == 1000] = 1

# Define the desired smoothing window in seconds
smoothing_window = 0.01  # Example: 10 ms

# Calculate sigma in terms of the number of samples
sigma = smoothing_window / dt

# Smooth the data using the calculated sigma
neural_data_filt = gaussian_filter1d(neural_data, sigma=sigma, axis=0)

# Create a figure with three subplots
fig = plt.figure(figsize=(11, 9), dpi=700)

# Plot each trace as black dots in the first subplot (big raster plot)
ax1 = fig.add_subplot(2, 1, 1)
for i in range(100):
    spike_times = sim.trange()[neural_data[:, i] > 0]
    ax1.scatter(spike_times, np.full_like(spike_times, i), color='black', s=0.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Neuron index')
ax1.set_title('Neural Activity Raster (first 100 neurons)')
ax1.set_xlim(0, T)  # Limit x-axis from 0 to T
ax1.set_ylim(0, 100)  # Limit x-axis from 0 to T

# Perform PCA on the time dimension of the raster data
pca = PCA(n_components=3)
pca_result = pca.fit_transform(neural_data_filt)

# Plot PC1 vs. time, PC2 vs. time, and PC3 vs. time in the second subplot
ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(sim.trange(), pca_result[:, 0], label='PC1', color='#000000', linewidth=5)  # Pure Black
ax2.plot(sim.trange(), pca_result[:, 1], label='PC2', color='#4F4F4F', linewidth=5)  # Dark Gray
ax2.plot(sim.trange(), pca_result[:, 2], label='PC3', color='#8F8F8F', linewidth=5)  # Medium Gray
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude, a.u.')
ax2.set_title('Projections onto first 3 Principal Components')
ax2.legend()
ax2.set_xlim(0, T)  # Limit x-axis from 0 to T


# Plot PC1 vs. PC2 vs. PC3 in the third subplot (3D plot) with color corresponding to time
ax3 = fig.add_subplot(2, 2, 4, projection='3d')

# Use the time vector (sim.trange()) as the color, with a grayscale colormap
sc = ax3.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
                 c=sim.trange(), cmap='RdPu', linewidth=2)

# Remove the gray grid background
ax3.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
ax3.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
ax3.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

ax3.set_xlabel('PC 1, a.u.')
ax3.set_ylabel('PC 2, a.u.')
ax3.set_zlabel('PC 3, a.u.')
ax3.set_title('Principal Component Space')

plt.show()

