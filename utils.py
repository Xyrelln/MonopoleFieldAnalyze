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


def create_filled_circle(center, radius, space_shape):
    cx, cy, cz = center
    # Prepare lists to store coordinates
    x_coords = []
    y_coords = []
    z_coords = []

    # Iterate over a grid that bounds the circle
    for x in range(cx - radius, cx + radius + 1):
        for y in range(cy - radius, cy + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(cz)  # z-coordinates remain constant

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)

    mask = (x_coords >= 0) & (x_coords < space_shape[0]) & \
           (y_coords >= 0) & (y_coords < space_shape[1]) & \
           (z_coords >= 0) & (z_coords < space_shape[2])

    return x_coords[mask], y_coords[mask], z_coords[mask]

def cal_interference(field_exp, center, radius, space_shape):
    '''
    Calculate the curent setup's interference with an imaginary anti wave with field flipped by x-axis
    :param field_exp: Field instance to be calculated
    :param center: observer center coordinates
    :param radius: observer radius
    :param space_shape: shape of field instance
    :return: complex sum observed, complex raw matrix observed, complex positive field observed, complex negative field observed
    '''
    x_coords, y_coords, z_coords = create_filled_circle(center, radius, space_shape)
    result_cache = np.zeros((radius*2+1, radius*2+1), dtype=complex)
    inverted_result_cache = np.zeros((radius*2+1, radius*2+1), dtype=complex)
    for i in tqdm(range(len(x_coords))):
        single = field_exp.cal_sound_pressure(x_coords[i], y_coords[i], z_coords[i])
        result_cache[x_coords[i] - (center[0]-radius)][y_coords[i] - (center[1]-radius)] = single
        single_inverted = field_exp.cal_sound_pressure(x_coords[i], y_coords[i], z_coords[i], np.pi)
        inverted_result_cache[x_coords[i] - (center[0]-radius)][y_coords[i] - (center[1]-radius)] = single_inverted
    no_int = result_cache  # not interfered positive wave
    no_int_inverted = inverted_result_cache  # not interfered negative wave
    no_int_inverted = np.flip(no_int_inverted, axis=0)  # flip by x-axis
    result = np.sum(no_int + no_int_inverted)  # complex sum
    result_cache = no_int + no_int_inverted  # complex field
    return result, result_cache, no_int, no_int_inverted