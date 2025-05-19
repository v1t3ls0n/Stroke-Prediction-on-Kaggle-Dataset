"""
Pure-Python C4.5 / J48-like decision-tree

• Gain-ratio for splits (categorical & numeric thresholds)
• Pessimistic-error post-pruning
• NetworkX graph visualisation + per-node gain-tables
• Optional tqdm progress-bar to monitor node construction
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any, Optional

# ────────────────────────── internal node classes ──────────────────────


class _Leaf:
    """Terminal node."""

    def __init__(self, label: int, proba: np.ndarray):
        self.label: int = label
        self.proba: np.ndarray = proba          # shape (n_classes,)


class _Internal:
    """Non-terminal node containing children."""

    def __init__(self, attr: str, thr: float | None = None):
        self.attr: str = attr
        self.thr:  float | None = thr          # None ⇒ categorical
        self.child: Dict[Any, Any] = {}         # key → node


# ───────────────────────────── C4.5 class ──────────────────────────────
class C45Classifier:
    """
    Parameters
    ----------
    min_samples_leaf : int
        Minimum samples in any leaf.
    pruning : bool
        Enable pessimistic-error pruning.
    show_tables : bool
        If True, keep full subsets (for `display_subtables`).
    progress_bar : tqdm.tqdm | None
        Pass an existing tqdm progress-bar instance to watch training
        progress. Each created node (leaf OR internal) increments by 1.
    """

    def __init__(self,
                 min_samples_leaf: int = 2,
                 pruning: bool = True,
                 show_tables: bool = False,
                 progress_bar: Optional[Any] = None):
        self.min_leaf = max(1, min_samples_leaf)
        self.pruning = pruning
        self.show_tbl = show_tables
        self.pbar = progress_bar          # can be None
        # internals
        self.tree = nx.DiGraph()
        # path-tuple → (df, best_attr, gain_df)
        self.subtables = {}
        self.classes_:  np.ndarray | None = None
        self.root_:     _Leaf | _Internal | None = None

    # ───────── entropy helpers ─────────────────────────────────────────
    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        cnt = np.bincount(y)
        p = cnt[cnt > 0] / cnt.sum()
        return -np.sum(p * np.log2(p))

    @staticmethod
    def _split_info(counts: np.ndarray) -> float:
        p = counts[counts > 0] / counts.sum()
        return -np.sum(p * np.log2(p))

    # ───────── gain-ratio functions ────────────────────────────────────
    def _gain_ratio_cat(self, X: pd.DataFrame, y: np.ndarray, attr: str) -> float:
        vals, counts = np.unique(X[attr], return_counts=True)
        cond_ent = sum((cnt / y.size) * self._entropy(y[X[attr] == v])
                       for v, cnt in zip(vals, counts))
        info_gain = self._entropy(y) - cond_ent
        split_info = self._split_info(counts)
        return 0 if split_info == 0 else info_gain / split_info

    def _best_split_num(self, X: pd.DataFrame, y: np.ndarray,
                        attr: str) -> Tuple[float, float | None]:
        """Return best gain-ratio and threshold for numeric attr."""
        order = np.argsort(X[attr].values)
        xs, ys = X[attr].values[order], y[order]
        change_idx = np.where(ys[:-1] != ys[1:])[0]
        best_gr, best_thr = -np.inf, None
        for i in change_idx:
            thr = (xs[i] + xs[i + 1]) / 2.0
            left, right = y[xs <= thr], y[xs > thr]
            if min(len(left), len(right)) < self.min_leaf:
                continue
            info_gain = (self._entropy(y) -
                         len(left) / y.size * self._entropy(left) -
                         len(right) / y.size * self._entropy(right))
            split_info = self._split_info(np.array([len(left), len(right)]))
            gr = 0 if split_info == 0 else info_gain / split_info
            if gr > best_gr:
                best_gr, best_thr = gr, thr
        return best_gr, best_thr

    # ───────── recursive builder ───────────────────────────────────────
    def _build(self, X: pd.DataFrame, y: np.ndarray,
               path: List[Tuple[str, str]]):
        # stop → leaf
        if len(np.unique(y)) == 1 or y.size < 2 * self.min_leaf or X.empty:
            leaf = _Leaf(np.bincount(y).argmax(), self._leaf_proba(y))
            if self.pbar is not None:
                self.pbar.update(1)
            return leaf

        best_attr, best_thr, best_gr = None, None, -np.inf
        gain_records = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                gr, thr = self._best_split_num(X, y, col)
            else:
                gr, thr = self._gain_ratio_cat(X, y, col), None
            gain_records.append({
                'Attribute': f"{col}" if thr is None else f"{col} ≤ {thr:.3g}",
                'Gain Ratio': gr})
            if gr > best_gr:
                best_attr, best_thr, best_gr = col, thr, gr

        if best_gr == -np.inf or best_attr is None:
            leaf = _Leaf(np.bincount(y).argmax(), self._leaf_proba(y))
            if self.pbar is not None:
                self.pbar.update(1)
            return leaf

        # log subset for display
        self.subtables[tuple(path)] = (
            X if self.show_tbl else X.iloc[:0],
            best_attr,
            pd.DataFrame(gain_records).sort_values(
                "Gain Ratio", ascending=False)
        )

        node = _Internal(best_attr, best_thr)
        if self.pbar is not None:
            self.pbar.update(1)

        node_id = self._node_id(path)          # e.g. ROOT or 'attr=val → ...'
        self.tree.add_node(
            node_id,
            label=node.attr if best_thr is None else f"{best_attr} ≤ {best_thr:.3g}",
            leaf=False, path=path.copy()
        )
        if path:
            self.tree.add_edge(self._node_id(
                path[:-1]), node_id, label=path[-1][1])

        # recurse
        if best_thr is None:   # categorical
            for v in X[best_attr].unique():
                mask = X[best_attr] == v
                node.child[v] = self._build(
                    X[mask].drop(columns=[best_attr]), y[mask],
                    path + [(best_attr, str(v))]
                )
        else:                  # numeric
            le_mask = X[best_attr] <= best_thr
            node.child["le"] = self._build(
                X[le_mask], y[le_mask],
                path + [(best_attr, f"≤{best_thr:.3g}")]
            )
            node.child["gt"] = self._build(
                X[~le_mask], y[~le_mask],
                path + [(best_attr, f">{best_thr:.3g}")]
            )

        if self.pruning:
            node = self._pessimistic_prune(node, y)
        return node

    # ───────── pruning / leaf helpers ──────────────────────────────────
    def _leaf_proba(self, y: np.ndarray) -> np.ndarray:
        cnt = np.bincount(y, minlength=len(self.classes_))
        return cnt / cnt.sum()

    def _pessimistic_prune(self, node, y_parent):
        if isinstance(node, _Leaf):
            return node
        for k in list(node.child):
            node.child[k] = self._pessimistic_prune(node.child[k], y_parent)
        if all(isinstance(ch, _Leaf) for ch in node.child.values()):
            leaf_err = min(len(y_parent), len(y_parent) - y_parent.sum()) + 0.5
            subtree_err = sum((1 - max(ch.proba)) * len(y_parent)
                              for ch in node.child.values())
            if leaf_err <= subtree_err + 0.1:
                return _Leaf(np.bincount(y_parent).argmax(), self._leaf_proba(y_parent))
        return node

    # ───────── public API ──────────────────────────────────────────────
    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).astype(int).values
        self.classes_ = np.unique(y)
        self.tree.clear()
        self.subtables.clear()
        self.root_ = self._build(X, y, [])
        return self

    def _traverse(self, row: pd.Series, node):
        while isinstance(node, _Internal):
            key = ("le" if row[node.attr] <= node.thr else "gt") if node.thr is not None \
                else row[node.attr]
            node = node.child.get(key)
            if node is None:   # unseen category
                return _Leaf(self.classes_[0],
                             np.ones(len(self.classes_)) / len(self.classes_))
        return node

    def predict(self, X):
        X = pd.DataFrame(X)
        return np.array([self._traverse(r, self.root_).label
                         for _, r in X.iterrows()])

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        return np.vstack([self._traverse(r, self.root_).proba
                          for _, r in X.iterrows()])

    # ───────── visualisation helpers ───────────────────────────────────
    def _node_id(self, path):              # unique string for each node
        return "ROOT" if not path else " → ".join(f"{a}={v}" for a, v in path)

    def _tree_depth(self, node_id):
        """Recursively compute the depth of a node for auto-sizing the layout."""
        children = list(self.tree.successors(node_id))
        if not children:
            return 1
        return 1 + max(self._tree_depth(child) for child in children)






    def visualize(self, figsize=None, save_path="."):
        if self.root_ is None:
            print("⚠️  Fit the tree first.")
            return

        root_id = "ROOT"
        n_nodes = len(self.tree.nodes)
        depth = self._tree_depth(root_id)

        # Auto-size if not provided
        if figsize is None:
            width = max(12, n_nodes // 2.5)
            height = max(6, depth * 1.2)
            figsize = (width, height)

        pos = self._hierarchy_pos(root_id, width=1.5, vert_gap=1.0)
        fig = plt.figure(figsize=figsize)
        nx.draw(self.tree, pos, with_labels=False, node_size=1400,
                node_color="lightgreen", edge_color="gray", font_size=8)
        nx.draw_networkx_labels(
            self.tree, pos,
            labels={n: self.tree.nodes[n]["label"] for n in self.tree},
            font_size=9)
        nx.draw_networkx_edge_labels(
            self.tree, pos,
            edge_labels=nx.get_edge_attributes(self.tree, 'label'),
            font_size=8)
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            print(f"📁 Saving tree plot to {save_path}")
            fig.savefig(save_path, dpi=200, bbox_inches='tight')

        plt.show()




    def _hierarchy_pos(self, root, width=1.0, vert_gap=1.0, vert_loc=0, xcenter=0.5):
        """Recurse to calculate hierarchical layout positions."""
        pos = {}

        def recurse(node, x, y, dx):
            pos[node] = (x, y)
            kids = list(self.tree.successors(node))
            if kids:
                step = dx / len(kids)
                for i, ch in enumerate(kids):
                    recurse(ch, x - dx/2 + step*(i + 0.5), y - vert_gap, step)

        recurse(root, xcenter, vert_loc, width)
        return pos

    # ───────── subset display ─────────────────────────────────────────
    def display_subtables(self, style_fn=None):
        from IPython.display import display, HTML
        display(HTML("<h2>📋 C4.5 Training Subsets</h2>"))
        for path, (df, best_attr, gain_df) in self.subtables.items():
            subset = 'ROOT' if not path else ' & '.join(
                f"{a}={v}" for a, v in path)
            display(HTML(f"<h4>Subset: {subset} (n={len(df)})</h4>"))
            if self.show_tbl:
                display(df.head())
            display(HTML("<strong>Gain-Ratio table:</strong>"))
            if style_fn:
                display(style_fn(gain_df, highlight_attr=best_attr))
            else:
                display(gain_df)


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