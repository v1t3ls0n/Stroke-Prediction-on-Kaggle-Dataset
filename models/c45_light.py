import numpy as np
import pandas as pd
from math import log2
from collections import defaultdict

class _Leaf:
    def __init__(self, label, proba):
        self.label = label
        self.proba = proba

    def predict(self, x): return self.label
    def predict_proba(self, x): return self.proba

class _Internal:
    def __init__(self, attr, thr=None):
        self.attr = attr
        self.thr = thr
        self.child = {}

    def route(self, x):
        val = x[self.attr]
        if self.thr is None:
            return self.child.get(val, None)
        return self.child["le"] if val <= self.thr else self.child["gt"]

class C45Classifier_Light:
    def __init__(self, min_samples_leaf=2, pruning=True, progress_bar=None):
        self.min_leaf = max(1, min_samples_leaf)
        self.pruning = pruning
        self.root_ = None
        self.classes_ = None
        self.pbar = progress_bar  # tqdm instance or None

    @staticmethod
    def _entropy(y):
        cnt = np.bincount(y)
        p = cnt[cnt > 0] / cnt.sum()
        return -np.sum(p * np.log2(p))

    @staticmethod
    def _split_info(w):
        p = w[w > 0] / w.sum()
        return -np.sum(p * np.log2(p))

    def _gain_ratio_cat(self, X, y, col):
        vals, counts = np.unique(X[col], return_counts=True)
        cond_ent = sum((counts[i] / y.size) * self._entropy(y[X[col] == val]) for i, val in enumerate(vals))
        info_gain = self._entropy(y) - cond_ent
        split_info = self._split_info(counts)
        return 0 if split_info == 0 else info_gain / split_info

    def _gain_ratio_num(self, X, y, col):
        order = np.argsort(X[col].values)
        x_sorted = X[col].values[order]
        y_sorted = y[order]
        mask = y_sorted[:-1] != y_sorted[1:]
        if not mask.any():
            return -np.inf, None
        thr_idx = np.where(mask)[0]
        best_gr, best_thr = -np.inf, None
        for idx in thr_idx:
            thr = (x_sorted[idx] + x_sorted[idx+1]) / 2.0
            left_mask = X[col] <= thr
            right_mask = ~left_mask
            wl, wr = left_mask.sum(), right_mask.sum()
            if min(wl, wr) < self.min_leaf:
                continue
            info_gain = (self._entropy(y) -
                         (wl / y.size) * self._entropy(y[left_mask]) -
                         (wr / y.size) * self._entropy(y[right_mask]))
            split_info = self._split_info(np.array([wl, wr]))
            gr = 0 if split_info == 0 else info_gain / split_info
            if gr > best_gr:
                best_gr, best_thr = gr, thr
        return best_gr, best_thr

    def _build(self, X, y):
        if self.pbar:
            self.pbar.update(1)

        if len(np.unique(y)) == 1 or X.empty or y.size < 2 * self.min_leaf:
            return _Leaf(np.bincount(y).argmax(), self._leaf_proba(y))

        best_attr, best_thr, best_gr = None, None, -np.inf
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                gr, thr = self._gain_ratio_num(X, y, col)
            else:
                gr, thr = self._gain_ratio_cat(X, y, col), None
            if gr > best_gr:
                best_attr, best_thr, best_gr = col, thr, gr

        if best_attr is None:
            return _Leaf(np.bincount(y).argmax(), self._leaf_proba(y))

        node = _Internal(best_attr, best_thr)

        if best_thr is None:
            for v in X[best_attr].unique():
                mask = X[best_attr] == v
                node.child[v] = self._build(X[mask].drop(columns=[best_attr]), y[mask])
        else:
            le_mask = X[best_attr] <= best_thr
            gt_mask = ~le_mask
            node.child["le"] = self._build(X[le_mask], y[le_mask])
            node.child["gt"] = self._build(X[gt_mask], y[gt_mask])

        return self._prune(node, y) if self.pruning else node

    def _leaf_proba(self, y):
        cnt = np.bincount(y, minlength=len(self.classes_))
        return cnt / cnt.sum()

    def _prune(self, node, y_parent):
        if isinstance(node, _Leaf):
            return node
        for k in list(node.child):
            node.child[k] = self._prune(node.child[k], y_parent)
        if all(isinstance(ch, _Leaf) for ch in node.child.values()):
            n_total = len(y_parent)
            subtree_err = sum(len(y_parent) * (1 - max(ch.proba)) for ch in node.child.values())
            leaf_err = (min(n_total, n_total - y_parent.sum()) + 0.5)
            if leaf_err <= subtree_err + 0.1:
                majority = max(node.child.values(), key=lambda c: c.proba.max()).label
                return _Leaf(majority, self._leaf_proba(y_parent))
        return node

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True).astype(int).values
        self.classes_ = np.unique(y)
        self.root_ = self._build(X.copy(), y)
        return self

    def _traverse(self, x, node):
        while isinstance(node, _Internal):
            node = node.route(x)
            if node is None:
                return self.classes_[0]
        return node.label

    def predict(self, X):
        X = pd.DataFrame(X)
        return np.array([self._traverse(row, self.root_) for _, row in X.iterrows()])

    
    def predict_proba(self, X):
        X = pd.DataFrame(X)
        out = []
        for _, row in X.iterrows():
            node = self.root_
            while isinstance(node, _Internal):
                node = node.route(row)
                if node is None:
                    node = _Leaf(self.classes_[0],
                                 np.ones(len(self.classes_)) / len(self.classes_))
                    break
            out.append(node.proba)
        return np.vstack(out)

    def get_params(self, deep=True):
        return {
            "min_samples_leaf": self.min_leaf,
            "pruning": self.pruning,
            "progress_bar": self.pbar
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key == "min_samples_leaf":
                self.min_leaf = value
            elif key == "pruning":
                self.pruning = value
            elif key == "progress_bar":
                self.pbar = value
        return self