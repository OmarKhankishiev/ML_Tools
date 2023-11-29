import numpy as np


class Conv1d:
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding='same',
            activation='relu'
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation

        self.W, self.biases = self.init_weight_matrix()

    def init_weight_matrix(self):
        np.random.seed(1)
        W = np.random.uniform(
                size=(self.in_channels, self.kernel_size, self.out_channels)
        )
        biases = np.random.uniform(size=(1, self.out_channels))
        return W, biases

    def forward(self, x):
        T = len(x[0])
        padded_input = self.pad_input(x)

        output = np.zeros((self.out_channels, T))

        for t in range(T):
            for c_out in range(self.out_channels):
                conv_result = np.sum(
                        self.W[:, :, c_out] * padded_input[:, t:t+self.kernel_size]
                )
                output[c_out, t] = conv_result + self.biases[0, c_out]

        if self.activation == 'relu':
            output = np.maximum(0, output)

        return output

    def pad_input(self, x):
        if self.padding == 'same':
            pad_width = ((0, 0), (self.kernel_size // 2, self.kernel_size // 2))
            padded_input = np.pad(x, pad_width, mode='constant', constant_values=0)
        else:
            padded_input = x
        return padded_input


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    in_channels, out_channels, kernel_size = map(int, input().split())
    input_vectors = read_matrix(in_channels)

    conv = Conv1d(in_channels, out_channels, kernel_size)
    output = conv.forward(input_vectors).round(3)
    print_matrix(output)


solution()
