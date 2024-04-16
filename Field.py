import numpy as np
import matplotlib.pyplot as plt


class Field:
    def __init__(self, size=(256, 256, 256), sound_source=None, c=343, Q=1):
        self.field = np.zeros(size)  # (x, y, z), value is f
        self.size = size
        self.c = c
        self.Q = Q
        self.add_sound_source(sound_source)

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

    def sound_pressure_single(self, x0, y0, z0, x, y, z, Q=None, f=None, c=None):
        k = 2 * np.pi * f / c  # wave number
        # Calculate the distance R from the source to the receiver
        R = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
        if R == 0:
            R = 1e-10  # a small number to avoid division by zero
        p = Q * np.exp(-1j * k * R) / (4 * np.pi * R)  # Calculate the sound pressure (complex-valued)
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

    def cal_sound_pressure(self, xr, yr, zr):
        affective_sources = np.nonzero(self.field)
        total = 0 + 0j
        for i in range(affective_sources[0].shape[0]):
            total += self.sound_pressure_single(
                affective_sources[0][i], affective_sources[1][i], affective_sources[2][i],
                xr, yr, zr, self.Q,
                self.field[affective_sources[0][i], affective_sources[1][i], affective_sources[2][i]],
                self.c)
        return total

    def show_field(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Extract non-zero elements for 3D plotting
        x, y, z = np.nonzero(self.field)
        f = self.field[x, y, z]  # Extracting frequency values at non-zero points

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
