import numpy as np
import cv2


class GaussManager:
    def __init__(self):
        self.video_channels = 3
        self.video_frame_rate = 60
        self.levels = 3
        self.min_frequency = 0.8
        self.max_frequency = 1.3
        self.buffer_size = 150
        self.buffer_index = 0

        self.fourier_transform_avg = np.zeros((self.buffer_size))

        self.frequencies = (1.0 * self.video_frame_rate) * np.arange(self.buffer_size) / (1.0 * self.buffer_size)

        self.mask = (self.frequencies > self.min_frequency) & (self.frequencies <= self.max_frequency)

        self.bpm_calculation_frequency = 15
        self.bpm_buffer_index = 0
        self.bpm_buffer_size = 10
        self.bpm_buffer = np.zeros((self.bpm_buffer_size))

        self.i = 0

    def buildGauss(self, frame, levels):
        pyramid = [frame]

        for _ in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)

        return pyramid

    def getBpm(self, frame):
        self.first_frame = np.zeros((frame.shape[0], frame.shape[1], self.video_channels))

        self.first_gauss = self.buildGauss(self.first_frame, self.levels + 1)[self.levels]

        self.video_gauss = np.zeros(
            (self.buffer_size, self.first_gauss.shape[0], self.first_gauss.shape[1], self.video_channels)
        )

        self.video_gauss[self.buffer_index] = self.buildGauss(frame, self.levels + 1)[self.levels]

        fourier_transform = np.fft.fft(self.video_gauss, axis=0)

        fourier_transform[self.mask == False] = 0

        if self.buffer_index % self.bpm_calculation_frequency == 0:
            self.i = self.i + 1

        for buf in range(self.buffer_size):
            self.fourier_transform_avg[buf] = np.real(fourier_transform[buf]).mean()

        hz = self.frequencies[np.argmax(self.fourier_transform_avg)]

        bpm = 60.0 * hz

        self.bpm_buffer[self.bpm_buffer_index] = bpm
        self.bpm_buffer_index = (self.bpm_buffer_index + self.i) % self.bpm_buffer_size
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size

        return self.bpm_buffer.mean()
