import numpy as np

class MyDecisionTreeRegressor:

    def __init__(self, max_depth=None, max_features=None, min_leaf_samples=1):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_leaf_samples = min_leaf_samples
        self._node = {
            'left': None,
            'right': None,
            'feature': None,
            'threshold': None,
            'depth': 0,
            'value': None
        }
        self.tree = None  # словарь, в котором будет храниться построенное дерево

    def fit(self, X, y):
        self.tree = {'root': self._node.copy()}  # создаём первую узел в дереве
        self._build_tree(self.tree['root'], X, y)  # запускаем рекурсивную функцию для построения дерева
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            pred_for_x = self._get_predict(self.tree['root'], x)  # рекурсивно ищем лист в дереве соответствующий объекту
            predictions.append(pred_for_x)
        return np.array(predictions)

    def decision_stump(self, X, y):
        best_Q = 0  
        best_j = None
        best_t = None
        best_left_ids = None
        best_right_ids = None
        y_preds_left = None
        y_preds_right = None

        for j in range(X.shape[1]):
            n = X.shape[0]
            cur_MSE = self.mean_squared_error(y)
            unique_j = np.unique(X[:, j])
            for t in range(len(unique_j) - 1):
                cur_t = (unique_j[t] + unique_j[t + 1]) / 2
                y_l = y[X[:, j] <= cur_t]
                y_r = y[X[:, j] > cur_t]
                num_l = len(y_l)
                if num_l == 0 or num_l == n:
                    continue
                cur_Q = cur_MSE - ((num_l / n) * self.mean_squared_error(y_l) + ((n - num_l) / n) * self.mean_squared_error(y_r))
                if cur_Q > best_Q:
                    best_Q = cur_Q
                    best_t = cur_t
                    best_j = j
                    best_left_ids = X[:, j] <= cur_t
                    best_right_ids = X[:, j] > cur_t
                    y_preds_left = np.mean(y_l)
                    y_preds_right = np.mean(y_r)

        result = [
            best_Q,
            best_j,
            best_t,
            best_left_ids,
            best_right_ids,
            y_preds_left,
            y_preds_right
        ]
        return result

    def get_best_split(self, X, y):
        best_split = self.decision_stump(X, y)
        best_j = best_split[1]
        best_t = best_split[2]
        best_left_ids = best_split[3]
        best_right_ids = best_split[4]

        return int(best_j), best_t, best_left_ids, best_right_ids

    def calc_Q(self, y, y_left, y_right):
        return self.mean_squared_error(y) - ((len(y_left) / len(y)) * self.mean_squared_error(y_left) 
                               + (len(y_right) / len(y)) * self.mean_squared_error(y_right))

    def mean_squared_error(self, y):
        return np.mean((y - np.mean(y))**2)

    def _build_tree(self, curr_node, X, y):
        if curr_node['depth'] == self.max_depth:
            curr_node['value'] = np.mean(y)
            return

        if len(np.unique(y)) == 1:
            curr_node['value'] = np.mean(y)
            return

        j, t, left_ids, right_ids = self.get_best_split(X, y)
        curr_node['feature'] = j
        curr_node['threshold'] = t

        left = self._node.copy()
        right = self._node.copy()

        left['depth'] = curr_node['depth'] + 1
        right['depth'] = curr_node['depth'] + 1

        curr_node['left'] = left
        curr_node['right'] = right

        self._build_tree(left, X[left_ids], y[left_ids])
        self._build_tree(right, X[right_ids], y[right_ids])

    def _get_predict(self, node, x):
        if node['threshold'] is None:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._get_predict(node['left'], x)
        else:
            return self._get_predict(node['right'], x)


class MyGradientBoostingRegressor:

    def __init__(self, learning_rate, max_depth, max_features, n_estimators):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y):
        # Инициализация предсказаний ансамбля нулевым значением или средним
        #initial_prediction = np.mean(y)
        initial_prediction = np.zeros(y.shape[0])
        current_prediction = np.full_like(y, initial_prediction)

        for _ in range(self.n_estimators):
            # Вычисление градиента
            gradient = y - current_prediction

            # Создание нового базового дерева и обучение его антиградиенту
            tree = MyDecisionTreeRegressor(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X, gradient)  # Обучаем дерево антиградиенту
            self.trees.append(tree)

            # Обновление предсказаний ансамбля с учетом шага обучения (learning_rate)
            current_prediction += (self.learning_rate * tree.predict(X))

    def predict(self, X):
        # Суммируем предсказания базовых деревьев
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions

# Считывание входных данных
def read_matrix(n, dtype=float):
    matrix = np.array([list(map(dtype, input().split())) for _ in range(n)])
    return matrix

def read_input_matrices(n, m, k):
    X_train, y_train, X_test = read_matrix(n), read_matrix(n), read_matrix(k)
    return X_train, y_train.flatten(), X_test

def print_matrix(matrix):
    for row in matrix:
        print(row)

# Тестирование
def solution():
    n, m, k = map(int, input().split())
    X_train, y_train, X_test = read_input_matrices(n, m, k)

    gb = MyGradientBoostingRegressor(learning_rate=0.1, max_depth=3, max_features=m, n_estimators=80)
    gb.fit(X_train, y_train)

    predictions = gb.predict(X_test)
    print_matrix(predictions)

solution()

