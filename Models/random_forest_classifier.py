import numpy as np

class MyDecisionTreeClassifier:

    def __init__(self, max_depth=None, max_features=None, min_leaf_samples=None,
                 classes=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_leaf_samples = min_leaf_samples
        self._node = {
                        'left': None,
                        'right': None,
                        'feature': None,
                        'threshold': None,
                        'depth': 0,
                        'classes_proba': None
                    }
        self.tree = None  # словарь в котором будет храниться построенное дерево
        self.classes = classes  # список меток классов

    def fit(self, X, y):
        if self.classes is None:
            self.classes = np.unique(y)  
        self.tree = {'root': self._node.copy()}  # создаём первую узел в дереве
        self._build_tree(self.tree['root'], X, y)  # запускаем рекурсивную функцию для построения дерева
        return self

    def predict_proba(self, X):
        proba_preds = []
        for x in X:
            preds_for_x = self._get_predict(self.tree['root'], x)  # рекурсивно ищем лист в дереве соответствующий объекту
            proba_preds.append(preds_for_x)
        return np.array(proba_preds)

    def predict(self, X):
        proba_preds = self.predict_proba(X)
        preds = proba_preds.argmax(axis=1).reshape(-1, 1)
        return preds

    
    def get_best_split(self, X, y):
        q = 0
        best_j = 0
        best_t = 0
        n = X.shape[0]
        for attr in range(X.shape[1]):
            X_sorted = np.unique(X[:, attr])
            for i in range(X_sorted.shape[0] - 1):
                div = np.mean([X_sorted[i], X_sorted[i+1]])
                y_l = y[(X[:, attr].reshape(-1,1) < div)]
                y_r = y[(X[:, attr].reshape(-1,1) >= div)]
                cnt = len(y_l)
                if cnt == 0 or cnt == n:
                    continue
                
                Q = self.calc_Q(y, y_l, y_r)
                if Q > q:
                    q = Q
                    best_j = attr
                    best_t = div
        best_left_ids = np.where(X[:, best_j] < best_t)
        best_right_ids = np.where(X[:, best_j] >= best_t)

        return best_j, best_t, best_left_ids, best_right_ids

    def calc_Q(self, y, y_left, y_right):
        p_l = y_left.shape[0]/y.shape[0]
        p_r = y_right.shape[0]/y.shape[0]
        return self.gini(y) - (self.gini(y_left)*p_l + self.gini(y_right)*p_r)


    def gini(self, y):
        p_k = np.unique(y, return_counts=True)[1] / len(y)
        return np.sum(p_k * (1 - p_k))

    def _build_tree(self, curr_node, X, y):

        if curr_node['depth'] == self.max_depth:  # выход из рекурсии если построили до максимальной глубины
            curr_node['classes_proba'] = {c: (y == c).mean() for c in self.classes}  # сохраняем предсказания листьев дерева перед выходом из рекурсии
            return

        if len(np.unique(y)) == 1:  # выход из рекурсии значения если "y" одинковы для все объектов
            curr_node['classes_proba'] = {c: (y == c).mean() for c in self.classes}
            return

        j, t, left_ids, right_ids = self.get_best_split(X, y)  # нахождение лучшего разбиения

        curr_node['feature'] = j  # признак по которому производится разбиение в текущем узле
        curr_node['threshold'] = t  # порог по которому производится разбиение в текущем узле

        left = self._node.copy()  # создаём узел для левого поддерева
        right = self._node.copy()  # создаём узел для правого поддерева

        left['depth'] = curr_node['depth'] + 1  # увеличиваем значение глубины в узлах поддеревьев
        right['depth'] = curr_node['depth'] + 1

        curr_node['left'] = left
        curr_node['right'] = right

        self._build_tree(left, X[left_ids], y[left_ids])  # продолжаем построение дерева
        self._build_tree(right, X[right_ids], y[right_ids])
    
    def _get_predict(self, node, x):
        if node['threshold'] is None:  # если в узле нет порога, значит это лист, выходим из рекурсии
            return [node['classes_proba'][c] for c in self.classes]

        if x[node['feature']] <= node['threshold']:  # уходим в правое или левое поддерево в зависимости от порога и признака
            return self._get_predict(node['left'], x)
        else:
            return self._get_predict(node['right'], x)
        

class MyRandomForestClassifier:

    def __init__(self, max_features, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.classes = None

        
    def fit(self, X, y):
        
        self.forest = []
        self.classes = np.unique(y)
        self.n_classes  = len(self.classes)
        
        for _ in range(self.n_estimators):
            idx = np.random.choice(X.shape[0], X.shape[0])
            clr = MyDecisionTreeClassifier(max_features=self.max_features,
                                         max_depth=self.max_depth,
                                         classes = self.classes)
            clr.fit(X[idx], y[idx])
            self.forest.append(clr)
    
        return self
    
    def predict(self, X):
        preds = self.predict_proba(X)
        return np.argmax(preds, axis=1)
    
    def predict_proba(self, X):
        preds = np.mean(np.array([c.predict_proba(X) for c in self.forest]), axis=0)
        return preds

def read_matrix(n, dtype=float):
    matrix = np.array([list(map(dtype, input().split())) for _ in range(n)])
    return matrix

def read_input_matriсes(n, m, k):
    X_train, y_train, X_test = read_matrix(n), read_matrix(n), read_matrix(k)
    return X_train, y_train, X_test

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

def solution():
    n, m, k = map(int, input().split())
    X_train, y_train, X_test = read_input_matriсes(n, m, k)

    rf = MyRandomForestClassifier(max_features=4, n_estimators=100)    
    rf.fit(X_train, y_train)

    predictions = rf.predict_proba(X_test)
    print_matrix(predictions)

solution()

