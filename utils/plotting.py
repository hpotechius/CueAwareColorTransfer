"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# Plot the color distribution of an image as a 3D RGB scatter plot.

# Args:
#     image: RGB image as a NumPy array with shape (H, W, 3).
#     max_points: Maximum number of randomly sampled pixels (for performance).
#     filename: Optional filename used for informational output.
# ----------------------------------------------------------------------------------------------------------------------
def plot_color_distribution_3d(image, max_points=100000, filename="color_distribution_3d.png"):
    img = image.reshape(-1, 3)
    if len(img) > max_points:
        idx = np.random.choice(len(img), max_points, replace=False)
        img = img[idx]
    img = np.clip(img, 0, 1)
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(img[:,0], img[:,1], img[:,2], c=img, s=2, alpha=0.5, marker='o')
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    # Drop alpha channel if present
    if buf.shape[-1] == 4:
        plot_img = buf[...,:3].copy()
    else:
        plot_img = buf.copy()

    plt.close(fig)
    return plot_img