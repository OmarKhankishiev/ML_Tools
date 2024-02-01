import numpy as np

def mse(y):
    return np.mean((y - np.mean(y))**2)

def decision_stump(X, y):
    
    n, m = X.shape
    
    best_Q = float('-inf')  # начальное значение критерия информативности
    best_j = None  # индекс признака по которому производился лучший сплит
    best_t = None  # порог с котором сравнивается признак
    best_left_ids = None  # вектор со значениями True для объектов в левом поддереве, остальные False
    best_right_ids = None  # вектор со значениями True для объектов в правом поддереве, остальные False
    y_preds_left = None  # предсказание в левом поддерева
    y_preds_right = None  # предсказание в правом поддерева

    for j in range(m):
        unique_values = np.unique(X[:, j])
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        for t in thresholds:
            left_ids = X[:, j] <= t
            right_ids = ~left_ids

            #if left_ids.sum() == 0 or right_ids.sum() == 0:
            #    continue

            y_left = y[left_ids]
            y_right = y[right_ids]

            current_Q = mse(y) - (left_ids.sum() / n) * mse(y_left) - (right_ids.sum() / n) * mse(y_right)

            if current_Q > best_Q:
                best_Q = current_Q
                best_j = j
                best_t = t
                best_left_ids = left_ids
                best_right_ids = right_ids
                y_preds_left = np.mean(y_left)
                y_preds_right = np.mean(y_right)

    result = [
        best_Q,
        best_j,
        best_t,
        best_left_ids.sum(),
        best_right_ids.sum(),
        y_preds_left,
        y_preds_right
    ]
    return result

def read_input():
    n, m = map(int, input().split())
    x_train = np.array([input().split() for _ in range(n)]).astype(float)
    y_train = np.array([input().split() for _ in range(n)]).astype(float)
    return x_train, y_train

def solution():
    X, y = read_input()
    result = decision_stump(X, y)
    result = np.round(result, 2)
    output = ' '.join(map(str, result))
    print(output)

solution()

