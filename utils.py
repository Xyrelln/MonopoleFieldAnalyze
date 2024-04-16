import numpy as np
from tqdm import tqdm


def sample_arc_with_pressure(center, radius, space_shape, field):
    cx, cy, cz = center
    angles = np.linspace(0, np.pi, 1800)  # Semi-circle from 0 to 180 degrees
    pressures = []
    angle_values = []

    for theta in tqdm(angles):
        x = cx + int(radius * np.cos(theta))
        z = cz + int(radius * np.sin(theta))
        y = cy  # Keep y constant as the center's y

        pressure = np.abs(field.cal_sound_pressure(x, y, z))
        pressures.append(pressure)
        angle_values.append(theta)

    return angle_values, pressures


def sample_arc_with_pressure_ff(center, radius, space_shape, field):
    cx, cy, cz = center
    angles = np.linspace(0, np.pi, 1800)  # Semi-circle from 0 to 180 degrees
    pressures = []
    angle_values = []

    for theta in tqdm(angles):
        x = cx + int(radius * np.cos(theta))
        z = cz + int(radius * np.sin(theta))
        y = cy  # Keep y constant as the center's y

        if 0 <= x < space_shape[0] and 0 <= z < space_shape[2]:
            # Calculate sound pressure, assuming a function field.cal_sound_pressure exists
            pressure = np.abs(field.cal_sound_pressure_ff(x, y, z))
            pressures.append(pressure)
            angle_values.append(theta)

    return angle_values, pressures


def sample_hemisphere(center, radius, space_shape):
    '''
    Return an array with indexes of hemisphere. Used to index field.
    :return
    For returned[i]:
    returned[i][0] is x coordinate of this sampled point.
    Same for returned[i][1], returned[i][2] is y,z coordinate of this sampled point.
    '''
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


def sample_arc(center, radius, azimuth_angle, space_shape):
    cx, cy, cz = center
    space = np.zeros(space_shape, dtype=bool)

    # Only loop through half a circle for a semi-circle arc
    for theta in np.linspace(0, np.pi, 180):  # Semi-circle from 0 to 180 degrees
        x = cx + int(radius * np.cos(theta))
        z = cz + int(radius * np.sin(theta))
        y = cy  # Keep y constant as the center's y

        if 0 <= x < space_shape[0] and 0 <= y < space_shape[1] and 0 <= z < space_shape[2]:
            space[x, y, z] = True

    return np.transpose(np.nonzero(space))