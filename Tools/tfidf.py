import numpy as np


class TfidfVectorizer:
    def __init__(self):
        self.sorted_vocab = {}
        self.idf = {}

    def fit(self, X):
        document_count = len(X)
        term_frequency = {}

        for doc in X:
            terms = doc.split()
            unique_terms = set(terms)

            for term in unique_terms:
                term_frequency[term] = term_frequency.get(term, 0) + 1

        for term, freq in term_frequency.items():
            self.idf[term] = np.log(document_count / (freq))

        self.sorted_vocab = {term: idx for idx, (term, _) in enumerate(sorted(self.idf.items()))}

        return self

    def transform(self, X):
        matrix = np.zeros((len(X), len(self.sorted_vocab)))

        for i, doc in enumerate(X):
            terms = doc.split()

            for term in terms:
                if term in self.sorted_vocab:
                    term_idx = self.sorted_vocab[term]
                    tf = terms.count(term) / len(terms)
                    matrix[i, term_idx] = tf * self.idf[term]

        return matrix


def read_input():
    n1, n2 = map(int, input().split())

    train_texts = [input().strip() for _ in range(n1)]
    test_texts = [input().strip() for _ in range(n2)]

    return train_texts, test_texts


def solution():
    train_texts, test_texts = read_input()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_texts)
    transformed = vectorizer.transform(test_texts)

    for row in transformed:
        row_str = ' '.join(map(str, np.round(row, 3)))
        print(row_str)


solution()
