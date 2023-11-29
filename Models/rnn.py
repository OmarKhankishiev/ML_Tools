import numpy as np


class RNN:

    def __init__(self, in_features, hidden_size, n_classes, activation='tanh'):
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.activation = activation
        self.init_weights()

    def init_weight_matrix(self, size):
        np.random.seed(1)
        W = np.random.uniform(size=size).reshape(size)
        return W

    def init_weights(self):
        np.random.seed(1)
        self.W_ax = self.init_weight_matrix((self.hidden_size, self.in_features))
        self.W_aa = self.init_weight_matrix((self.hidden_size, self.hidden_size))
        self.W_ya = self.init_weight_matrix((self.n_classes, self.hidden_size))
        self.b_a = self.init_weight_matrix((self.hidden_size, 1))
        self.b_y = self.init_weight_matrix((self.n_classes, 1))

    def forward(self, x):
        x = np.transpose(x)
        a = np.zeros((self.hidden_size, 1))

        hidden_states = []

        for t in x:
            xt = t.reshape(-1, 1)
            step_a = np.dot(self.W_ax, xt) + np.dot(self.W_aa, a) + self.b_a
            a = np.tanh(step_a)
            hidden_states.append(a)

        outputs = [
                np.dot(
                    self.W_ya, hidden_state
                ) + self.b_y for hidden_state in hidden_states
        ]
        softmax_outputs = [self.softmax(output) for output in outputs]

        return np.transpose(np.array(softmax_outputs))[0]

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    in_features, hidden_size, n_classes = map(int, input().split())
    input_vectors = read_matrix(in_features)

    rnn = RNN(in_features, hidden_size, n_classes)
    output = rnn.forward(input_vectors).round(3)
    print_matrix(output)


solution()
