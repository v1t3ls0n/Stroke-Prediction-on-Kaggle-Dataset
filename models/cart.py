import numpy as np
import pandas as pd
import networkx as nx

class CARTDecisionTree:
    def __init__(self, min_samples_leaf=2):
        self.min_leaf = max(1, min_samples_leaf)
        self.tree = nx.DiGraph()
        self.root = None
        self.classes_ = None

    def _gini(self, y):
        p = np.bincount(y) / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        best_attr, best_thr, best_gini = None, None, float('inf')
        for attr in X.columns:
            if pd.api.types.is_numeric_dtype(X[attr]):
                values = X[attr].sort_values().unique()
                thresholds = (values[:-1] + values[1:]) / 2
                for thr in thresholds:
                    left, right = y[X[attr] <= thr], y[X[attr] > thr]
                    if len(left) < self.min_leaf or len(right) < self.min_leaf:
                        continue
                    gini = (len(left) * self._gini(left) + len(right) * self._gini(right)) / len(y)
                    if gini < best_gini:
                        best_attr, best_thr, best_gini = attr, thr, gini
            else:
                for val in X[attr].unique():
                    left, right = y[X[attr] == val], y[X[attr] != val]
                    if len(left) < self.min_leaf or len(right) < self.min_leaf:
                        continue
                    gini = (len(left) * self._gini(left) + len(right) * self._gini(right)) / len(y)
                    if gini < best_gini:
                        best_attr, best_thr, best_gini = attr, val, gini
        return best_attr, best_thr

    def _grow(self, X, y, path):
        if len(np.unique(y)) == 1 or len(y) < 2 * self.min_leaf:
            leaf_label = np.bincount(y).argmax()
            node = f"leaf:{leaf_label}-{len(self.tree.nodes)}"
            self.tree.add_node(node, label=leaf_label, leaf=True)
            return node

        attr, thr = self._best_split(X, y)
        if attr is None:
            leaf_label = np.bincount(y).argmax()
            node = f"leaf:{leaf_label}-{len(self.tree.nodes)}"
            self.tree.add_node(node, label=leaf_label, leaf=True)
            return node

        node = f"{attr}≤{thr}" if pd.api.types.is_numeric_dtype(X[attr]) else f"{attr}={thr}"
        self.tree.add_node(node, label=node, leaf=False)

        if pd.api.types.is_numeric_dtype(X[attr]):
            left_idx, right_idx = X[attr] <= thr, X[attr] > thr
            left = self._grow(X[left_idx], y[left_idx], path + [(attr, f"≤{thr:.2f}")])
            right = self._grow(X[right_idx], y[right_idx], path + [(attr, f">{thr:.2f}")])
            self.tree.add_edge(node, left, label=f"≤{thr:.2f}")
            self.tree.add_edge(node, right, label=f">{thr:.2f}")
        else:
            match_idx = X[attr] == thr
            left = self._grow(X[match_idx], y[match_idx], path + [(attr, str(thr))])
            right = self._grow(X[~match_idx], y[~match_idx], path + [(attr, f"≠{thr}")])
            self.tree.add_edge(node, left, label=str(thr))
            self.tree.add_edge(node, right, label=f"¬{thr}")

        return node

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).astype(int).reset_index(drop=True)
        self.classes_ = np.unique(y)
        self.tree.clear()
        self.root = self._grow(X, y, path=[])

    def _traverse(self, row, node):
        while not self.tree.nodes[node].get("leaf", False):
            label = self.tree.nodes[node]["label"]
            attr, op, val = self._parse_condition(label)
            if op == "≤":
                node = next((t for _, t, d in self.tree.out_edges(node, data=True)
                             if float(row[attr]) <= float(val) and "≤" in d["label"]), None)
            elif op == ">":
                node = next((t for _, t, d in self.tree.out_edges(node, data=True)
                             if float(row[attr]) > float(val) and ">" in d["label"]), None)
            elif op == "=":
                node = next((t for _, t, d in self.tree.out_edges(node, data=True)
                             if str(row[attr]) == val), None)
            else:
                node = None
            if node is None:
                return np.random.choice(self.classes_)
        return self.tree.nodes[node]["label"]

    def _parse_condition(self, label):
        if "≤" in label:
            return label.split("≤")[0], "≤", label.split("≤")[1]
        elif ">" in label:
            return label.split(">")[0], ">", label.split(">")[1]
        elif "=" in label:
            return label.split("=")[0], "=", label.split("=")[1]
        return label, "", ""

    def predict(self, X):
        return np.array([self._traverse(row, self.root) for _, row in pd.DataFrame(X).iterrows()])

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(preds), len(self.classes_)))
        for i, p in enumerate(preds):
            proba[i, p] = 1.0
        return proba
    
    def get_params(self, deep=True):
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self