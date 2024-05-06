import numpy as np
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm


class Field:
    def __init__(self, size=(256, 256, 256), sound_source=None, c=343, Q=1):
        self.field = np.zeros(size)  # (x, y, z), value is f
        self.size = size
        self.c = c
        self.Q = Q
        self.add_sound_source(sound_source)
        self.cache = dict()

    def add_sound_source(self, sound_sources):
        if sound_sources is None:
            return
        source_type = sound_sources[-1]
        if source_type == 'monopole':  # (x, y, z, f)
            for sound_source in sound_sources:
                if self.field[sound_source[:3]] != 0:
                    print(f'Warning: sound source at {sound_source[:3]} overwritten to frequency {sound_source[3]}')
                self.field[sound_source[:3]] = sound_source[3]

        if source_type == 'piston':  # ((x1, y1), (x2, y2), z, f)
            # only horizontal for now
            for x in range(sound_sources[0][0], sound_sources[1][0]):
                for y in range(sound_sources[0][1], sound_sources[1][1]):
                    self.field[x, y, sound_sources[2]] = sound_sources[3]

        if source_type == 'circular_piston':  # ((x_center, y_center), x, radius, f):
            x_center, y_center = sound_sources[0]
            z = sound_sources[1]
            radius = sound_sources[2]
            frequency = sound_sources[3]

            # Calculate the points within the circle
            for x in range(x_center - radius, x_center + radius + 1):
                for y in range(y_center - radius, y_center + radius + 1):
                    if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                        if self.field[x, y, z] != 0:
                            print(f'Warning: sound source at {(x, y, z)} overwritten to frequency {frequency}')
                        self.field[x, y, z] = frequency

    def sound_pressure_single(self, x0, y0, z0, x, y, z, phase=0, Q=None, f=None, c=None):
        if c is None:
            c = self.c
        k = 2 * np.pi * f / c  # wave number
        # Calculate the distance R from the source to the receiver
        R = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
        if R == 0:
            R = 1e-10  # a small number to avoid division by zero
        if Q is None:
            Q = self.Q
        p = Q * np.exp(-1j * (k * R + phase)) / (4 * np.pi * R)  # Calculate the sound pressure (complex-valued)
        return p

    def cal_sound_pressure_ff(self, xr, yr, zr, Q=None, f=None, c=None, far_field_center=None):
        '''
        Calculate the sound pressure at (xr, yr, zr).
        ASSUMING SOUND SOURCES ARE SINGLE FREQUENCIED
        f assigned to a random frequency in field if not specified
        :return: Complex pressure at observer point
        '''
        if Q is None:
            Q = self.Q
        if f is None:
            coord = np.nonzero(self.field)
            f = self.field[coord[0][0]][coord[1][0]][coord[2][0]]
        if c is None:
            c = self.c
        k = 2 * np.pi * f / c
        if far_field_center is None:  # if not specified center, use center of xy plane with z = 0
            far_field_center = (self.size[0] / 2, self.size[1] / 2, self.size[2] / 2)
            R = np.sqrt(
                (xr - far_field_center[0]) ** 2 + (yr - far_field_center[1]) ** 2 + (zr - far_field_center[2]) ** 2)
            if R == 0:
                R = 1e-10  # a small number to avoid division by zero
        p = Q * np.exp(-1j * k * R) / (4 * np.pi * R)
        total = p * len(np.nonzero(self.field))
        return total

    def add_cache(self, freq, z0, phase):
        sx, sy, sz = self.size
        curr_cache = np.zeros((sx*2-1, sy*2-1), dtype=complex)
        first_quarter = np.zeros((sx, sy), dtype=complex)

        if (freq, z0, phase) not in self.cache:
            for x in tqdm(range(self.size[0])):
                for y in range(self.size[1]):
                    first_quarter[x, y] = self.sound_pressure_single(x, y, z0,
                                                                        self.size[0]-1, self.size[1]-1, 0,
                                                                        freq, z0, phase)
            curr_cache[sx - 1:, :sy] = np.rot90(first_quarter, k=1)
            curr_cache[sx - 1:, sy - 1:] = np.rot90(first_quarter, k=2)
            curr_cache[:sx, sy - 1:] = np.rot90(first_quarter, k=-1)
            curr_cache[:sx, :sy] = first_quarter
        self.cache[(freq, z0, phase)] = curr_cache

    def cal_sound_pressure(self, xr, yr, zr, phase=0, build_cache=False):
        affective_sources = np.nonzero(self.field)
        x_offset = int(np.ceil(self.size[0]))
        y_offset = int(np.ceil(self.size[1]))
        total = 0 + 0j
        for i in range(affective_sources[0].shape[0]):
            x, y, z = affective_sources[0][i], affective_sources[1][i], affective_sources[2][i]
            # if (self.field[x, y, z], np.abs(zr - z), phase) in self.cache:
            #     total += self.cache[(self.field[x, y, z], np.abs(zr - z), phase)][x_offset+xr-x][y_offset+yr-y]
            # elif build_cache is True:
            #     print('build cache...')
            #     self.add_cache(self.field[x, y, z], np.abs(zr - z), phase)
            #     print('complete')
            #     total += self.cache[(self.field[x, y, z], np.abs(zr - z), phase)][x_offset+xr-x][y_offset+yr-y]
            # else:
            #     total += self.sound_pressure_single(
            #         x, y, z,
            #         xr, yr, zr, phase, self.Q,
            #         self.field[x, y, z],
            #         self.c)
            total += self.sound_pressure_single(
                x, y, z,
                xr, yr, zr, phase, self.Q,
                self.field[x, y, z],
                self.c)
        return total

    def show_field(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Extract non-zero elements for 3D plotting
        x, y, z = np.nonzero(self.field)
        f = self.field[x, y, z]

        # Filter out z = 0 for visualization
        mask = z >= 0
        x, y, z, f = x[mask], y[mask], z[mask], f[mask]

        sc = ax.scatter(x, y, z, c=f, cmap='viridis', marker='o')
        fig.colorbar(sc, ax=ax, label='Frequency')

        # Set limits for z-axis to show only z > 0
        ax.set_xlim(0, self.field.shape[0])
        ax.set_ylim(0, self.field.shape[1])
        ax.set_zlim(0, self.field.shape[2])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('3D Visualization of Sound Sources in the Field')
        plt.show()

    def cal_self_interference(self, center, radius, space_shape=None, build_cache=False):
        '''
        Calculate the curent field's interference with an imaginary anti wave with field flipped by x-axis
        :param center: observer center coordinates
        :param radius: observer radius
        :param space_shape: shape of field instance
        :param build_cache: build cache or not. Building cache takes a while but accelerates later calculations of a field
        :return: complex sum observed, complex raw matrix observed, complex positive field observed, complex negative field observed
        '''
        if space_shape is None:
            space_shape = self.size
        x_coords, y_coords, z_coords = utils.create_filled_circle(center, radius, space_shape)
        result_cache = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=complex)
        inverted_result_cache = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=complex)

        for i in tqdm(range(len(x_coords))):
            single = self.cal_sound_pressure(x_coords[i], y_coords[i], z_coords[i], build_cache=build_cache)
            result_cache[x_coords[i] - (center[0] - radius)][y_coords[i] - (center[1] - radius)] = single

            single_inverted = self.cal_sound_pressure(x_coords[i], y_coords[i], z_coords[i], np.pi, build_cache=build_cache)
            inverted_result_cache[x_coords[i] - (center[0] - radius)][
                y_coords[i] - (center[1] - radius)] = single_inverted

        no_int = result_cache  # not interfered positive wave
        no_int_inverted = inverted_result_cache  # not interfered negative wave
        no_int_inverted = np.flip(no_int_inverted, axis=0)  # flip by x-axis
        result = np.sum(no_int + no_int_inverted)  # complex sum
        result_cache = no_int + no_int_inverted  # complex field
        return result, result_cache, no_int, no_int_inverted

    def cal_cancellation(self, observer_center, observer_radius):
        result, result_cache, no_int, no_int_inverted = self.cal_self_interference(center=observer_center,
                                                                                   radius=observer_radius)
        cancellation_ratio = np.abs(result) / np.abs(np.sum(no_int))
        return result, result_cache, cancellation_ratio
