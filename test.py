import numpy as np


def sample_hemisphere(center, radius, space_shape):
    # Unpack center coordinates
    cx, cy, cz = center

    # Initialize the 3D space
    space = np.zeros(space_shape, dtype=bool)

    # Sample points on the hemisphere
    # Loop over azimuthal angle theta from 0 to 360 degrees and polar angle phi from 0 to 90 degrees
    for theta in np.linspace(0, 2 * np.pi, 100):  # 100 points for theta
        for phi in np.linspace(0, np.pi / 2, 50):  # 50 points for phi (only top half)
            # Convert spherical to Cartesian coordinates
            x = cx + int(radius * np.sin(phi) * np.cos(theta))
            y = cy + int(radius * np.sin(phi) * np.sin(theta))
            z = cz + int(radius * np.cos(phi))

            # Check if the point is within the bounds of the space
            if 0 <= x < space_shape[0] and 0 <= y < space_shape[1] and 0 <= z < space_shape[2]:
                space[x, y, z] = True

    return np.transpose(np.nonzero(space))  # Return the indices of non-zero elements as sample points


# Center at (128, 128, 0) and radius 40
center = (128, 128, 0)
radius = 40
space_shape = (256, 256, 256)

# Get sampled points on the hemisphere
sampled_points = sample_hemisphere(center, radius, space_shape)

# Print some of the sampled points (optional, just to verify)
print("Sampled points on the hemisphere:", sampled_points[100:120])  # Print first 10 points for brevity
