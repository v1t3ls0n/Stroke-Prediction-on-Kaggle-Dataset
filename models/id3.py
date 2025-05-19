def install_if_missing(pkgs_map):
    import sys, subprocess
    for pip_name, import_name in pkgs_map.items():
        if __import__('importlib').util.find_spec(import_name) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])



install_if_missing({
    'pandas': 'pandas',
    'seaborn': 'seaborn',
    'matplotlib': 'matplotlib',
    'scikit-learn': 'sklearn',
    'networkx': 'networkx',
    'ipython': 'IPython',
    'numpy': 'numpy'
})


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from io import StringIO
import networkx as nx
from IPython.display import display, HTML
import numpy as np


def style_dataframe(df, highlight_attr: str = None):
    """
    Styles a DataFrame for notebook display.
    - Replaces missing values (pd.NA, np.nan) with "<NA>" (as a string) for visibility.
    - Applies consistent visual formatting.
    - Highlights row where Attribute == highlight_attr, if specified.
    """
    # Force missing values to display as "<NA>" string (required before styling)
    df = df.copy().astype(object)
    df = df.where(df.notna(), "<NA>")  # ✅ Proper conversion

    # Display index starting at 1 for clarity
    df.index = pd.Index(range(1, len(df) + 1))

    # Styling logic
    styles = [
        {'selector': 'thead th', 'props': [('text-align', 'center'), ('padding', '8px')]},
        {'selector': 'tbody th', 'props': [('border-right', '2px solid black'), ('padding', '8px')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '8px')]},
    ]

    float_cols = df.select_dtypes(include='float').columns
    formatters = {col: '{:.20f}'.format for col in float_cols}

    styled = df.style.set_table_styles(styles).format(formatters)\
        .set_table_attributes("style='margin-bottom:50px; border-collapse: separate;'")

    if highlight_attr and 'Attribute' in df.columns:
        def highlight_best(row):
            return ['background-color: mediumseagreen; color: white; font-weight: bold' if row['Attribute'] == highlight_attr else '' for _ in row]
        styled = styled.apply(highlight_best, axis=1)

    return styled


class ID3DecisionTree:
    """
    Plain ID3 decision tree with *optional* pessimistic post-pruning.

    Parameters
    ----------
    min_samples_leaf : int   -   don’t split if one side would get < N rows
    pruning          : bool  -   enable pessimistic error pruning
    show_tables      : bool  -   show IG tables during fitting
    auto_visualize   : bool  -   draw tree automatically after fit
    """

    # ───────────────────────── constructor ──────────────────────────
    def __init__(self, *, filepath: str | None = None,
                 df: pd.DataFrame | None = None,
                 target: str | None = None,
                 min_samples_leaf: int = 2,
                 pruning: bool = True,
                 show_tables: bool = False,
                 auto_visualize: bool = False):
        self.min_leaf  = max(1, min_samples_leaf)
        self.pruning   = pruning
        self.show_tbl  = show_tables
        self.auto_vis  = auto_visualize

        if filepath:
            self.df, self.target = self._load_data(filepath)
        elif df is not None and target is not None:
            self.df, self.target = df.copy(), target
        else:
            self.df, self.target = None, None

        self.tree            = nx.DiGraph()
        self.gain_records    = []
        self.subtables: dict = {}


    def _load_data(self, filepath: str) -> pd.DataFrame:
        # Load CSV and strip column whitespace
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        target = df.columns[-1]
        return df, target



    # ───────────────────────── utilities ────────────────────────────
    @staticmethod
    def _entropy(labels: pd.Series | np.ndarray) -> float:
        values, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    def _information_gain(self, df: pd.DataFrame, attr: str, path):
        total_entropy = self._entropy(df[self.target])
        vals, counts  = np.unique(df[attr], return_counts=True)

        cond_entropy = sum(
            (cnt / len(df)) * self._entropy(df[df[attr] == v][self.target])
            for v, cnt in zip(vals, counts)
        )

        ig = total_entropy - cond_entropy
        self.gain_records.append({
            "Subset": " → ".join(f"{a}={v}" for a, v in path) or "ROOT",
            "Attribute": attr,
            "Entropy": total_entropy,
            "Conditional Entropy": cond_entropy,
            "Information Gain": ig
        })
        return ig



    # ───────────────────────── fitting ──────────────────────────────
    def fit(self):
        if self.df is None or self.target is None:
            raise ValueError("Pass either `filepath=` or (`df=`, `target=`)")
        attrs = [c for c in self.df.columns if c != self.target]
        self.tree.clear(); self.subtables.clear(); self.gain_records.clear()
        self._grow(self.df.copy(), attrs, path=[], parent=None, edge_lab="")
        if self.auto_vis:
            display(self.visualize())
            self.display_subtables()

    # Recursive growth ------------------------------------------------
    def _grow(self, df, attrs, path, parent, edge_lab):
        y = df[self.target]
        # stop criteria
        if (len(attrs) == 0 or len(np.unique(y)) == 1 or len(df) < 2*self.min_leaf):
            self._add_leaf(path, parent, edge_lab, y.mode()[0])
            return

        # choose best attribute
        ig_scores = {a: self._information_gain(df, a, path) for a in attrs}
        best_attr = max(ig_scores, key=ig_scores.get)

        # store IG table for GUI
        gain_df = (pd.DataFrame(self.gain_records)
                     .query("Subset == @(' → '.join(f'{a}={v}' for a,v in path) or 'ROOT')")
                     .sort_values("Information Gain", ascending=False))
        self.subtables[tuple(path)] = (df.head(), best_attr, gain_df)

        node_id = self._node_id(path)
        self.tree.add_node(node_id, label=best_attr, leaf=False, path=path.copy())
        if parent: self.tree.add_edge(parent, node_id, label=edge_lab)

        for v in np.unique(df[best_attr]):
            sub = df[df[best_attr] == v]
            if len(sub) == 0: continue
            self._grow(sub, [a for a in attrs if a != best_attr],
                       path + [(best_attr, v)], node_id, str(v))

        # post-prune this node
        if self.pruning:
            self._pessimistic_prune(node_id)

    # ───────────────────────── pruning ──────────────────────────────
    def _pessimistic_prune(self, node):
        kids = list(self.tree.successors(node))
        if not kids:     # leaf
            return
        # prune bottom-up
        for k in kids:
            self._pessimistic_prune(k)
        # if all children are leaves:
        if not all(self.tree.nodes[k]["leaf"] for k in kids):
            return
        labels = [self.tree.nodes[k]["label"] for k in kids]
        majority = max(set(labels), key=labels.count)
        subtree_err = sum(l != majority for l in labels)
        leaf_err    = min(subtree_err, len(labels) - subtree_err) + 0.5  # correction
        if leaf_err <= subtree_err + 0.1:
            for k in kids: self.tree.remove_node(k)
            self.tree.nodes[node].update({"leaf": True, "label": majority})


    # ───────────────────────── helpers / predict ────────────────────

    def predict(self, X):
        X = pd.DataFrame(X)
        out = []
        for _, row in X.iterrows():
            node = "ROOT"
            while not self.tree.nodes[node].get("leaf", False):
                attr = self.tree.nodes[node]["label"]
                val  = row[attr]
                # follow edge
                nxt = None
                for _, ch, d in self.tree.out_edges(node, data=True):
                    if d["label"] == str(val):
                        nxt = ch; break
                if nxt is None:      # unseen category
                    nxt = max((ch for _, ch in self.tree.out_edges(node)),
                              key=lambda c: self.tree.nodes[c]["label"])
                node = nxt
            out.append(self.tree.nodes[node]["label"])
        return np.array(out)


    def _id3(self, df, attrs, parent, edge_label, path):
        # Recursive ID3 algorithm to grow the decision tree
        target_vals = df[self.target]
        is_leaf = len(attrs) == 0 or len(np.unique(target_vals)) <= 1

        if is_leaf:
            # Stop condition: no attributes or pure node
            leaf_value = target_vals.mode()[0]
            entropy_val = self._entropy(target_vals)
            self.gain_records.append({
                'Subset': ' → '.join(f"{a}={v}" for a, v in path) or 'ROOT',
                'Attribute': '(Leaf Node)',
                'Entropy': abs(entropy_val),
                'Conditional Entropy (Remainder)': 'None',
                'Information Gain': 'None'
            })
            self.subsets_by_path[tuple(path)] = (df.copy(), None, None)
            if self.show_tables:
                display(HTML(f"→ Leaf: <strong>{leaf_value}</strong><hr>"))
            self._add_leaf(path, parent, edge_label, leaf_value)
            return

        # Compute IG for all attributes
        gains = []
        for attr in attrs:
            _, full_record = self._information_gain(df, attr, path)
            gains.append(full_record)

        gain_df = pd.DataFrame(gains).sort_values(by='Information Gain', ascending=False)
        best = max(gains, key=lambda x: x['Information Gain'])['Attribute']

        # Save current subset and best split for visualization
        self.subsets_by_path[tuple(path)] = (df.copy(), best, gain_df.copy())

        if self.show_tables:
            subset_name = 'ROOT' if not path else ' & '.join(f"{a}={v}" for a, v in path)
            display(HTML(f"<h4>▶ Subset: {subset_name} (n={len(df)})</h4>"))
            display(style_dataframe(df.copy(), highlight_attr=best))
            display(HTML("<h5>📊 Information Gain Table:</h5>"))
            display(style_dataframe(
                gain_df[['Attribute', 'Information Gain', 'Conditional Entropy (Remainder)', 'Entropy']],
                highlight_attr=best
            ))

        # Create internal node and recurse
        node_id = self._create_node_id(path)
        self.tree.add_node(node_id, label=best, leaf=False, path=path.copy())
        if parent:
            self.tree.add_edge(parent, node_id, label=edge_label)

        for val in np.unique(df[best]):
            subset = df[df[best] == val]
            if not subset.empty:
                self._id3(
                    subset,
                    [a for a in attrs if a != best],
                    node_id,
                    str(val),
                    path + [(best, val)]
                )

    def _add_leaf(self, path, parent, edge_label, value):
        # Create and add a leaf node to the tree
        cond = ' , '.join(f"{a}='{v}'" for a, v in path)
        node_id = f"🍃 Pr({self.target} | {cond})" if path else f"🍃 Pr({self.target})"
        self.tree.add_node(node_id, label=str(value), leaf=True, path=path.copy())
        if parent:
            self.tree.add_edge(parent, node_id, label=edge_label)

    def _create_node_id(self, path):
        # Generate a unique node name using icons and path
        cond = ' , '.join(f"{a}='{v}'" for a, v in path)
        prefix = f"Pr({self.target}" + (f" | {cond}" if cond else "") + ")"
        icon = "🪵" if not path else "🔷"
        return f"{icon} {prefix}"

    def display_subtables(self):
        # Show all training subsets and IG tables used during training
        display(HTML(f"<h1> 📋 Training Subset tables at Each Node </h1>"))
        for path, (df, best_attr, gain_df) in self.subsets_by_path.items():
            subset_name = 'ROOT' if not path else ' & '.join(f"{a}={v}" for a, v in path)
            display(HTML(f"<h2>▶ Subset: {subset_name} (n={len(df)})</h2>"))
            display(style_dataframe(df.copy(), highlight_attr=best_attr))
            if gain_df is not None:
                display(HTML("<h5>📊 Information Gain Table:</h5>"))
                display(style_dataframe(
                    gain_df[['Attribute', 'Information Gain', 'Conditional Entropy (Remainder)', 'Entropy']],
                    highlight_attr=best_attr
                ))

    def _filter(self, df, path):
        # Filter rows by path conditions
        for col, val in path:
            df = df[df[col] == val]
        return df.copy()

    def visualize(self, save_path=None, debug=False):
        # Visualize the built decision tree using networkx + matplotlib
        debug and print("🎯 Visualizing Tree")
        root = next((n for n, d in self.tree.in_degree() if d == 0), None)
        if root is None:
            print("❌ No root node found. Tree is empty.")
            return

        pos = self._hierarchy_pos(root)

        labels = nx.get_node_attributes(self.tree, 'label')
        nodes_data = dict(self.tree.nodes(data=True))

        diamonds = [root]
        internals = [n for n in self.tree.nodes if not nodes_data[n].get('leaf', False) and n != root]
        leaves = [n for n in self.tree.nodes if nodes_data[n].get('leaf', False)]
        leaf_colors = ['teal' if labels.get(n) == 'yes' else 'darkred' for n in leaves]

        fig = plt.figure(figsize=(18, 10))
        nx.draw_networkx_nodes(self.tree, pos, nodelist=diamonds, node_shape='D', node_color='skyblue', node_size=4000)
        nx.draw_networkx_nodes(self.tree, pos, nodelist=internals, node_shape='D', node_color='lightgreen', node_size=3500)
        nx.draw_networkx_nodes(self.tree, pos, nodelist=leaves, node_shape='o', node_color=leaf_colors, node_size=3000)
        nx.draw_networkx_edges(self.tree, pos)
        nx.draw_networkx_edge_labels(self.tree, pos, edge_labels=nx.get_edge_attributes(self.tree, 'label'), font_color='blue', font_size=14)
        nx.draw_networkx_labels(self.tree, pos, labels, font_size=12)

        plt.axis('off')
        plt.margins(0.1)

        if save_path:
            print(f"💾 Saving tree plot to: {save_path}")
            fig.savefig(save_path, dpi=200, bbox_inches='tight')

        plt.show()

    def _hierarchy_pos(self, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        # Assign 2D positions to each node for tree-like plotting
        pos = {}
        def recurse(node, x, y, dx):
            pos[node] = (x, y)
            children = list(self.tree.successors(node))
            if children:
                step = dx / len(children)
                for i, child in enumerate(children):
                    recurse(child, x - dx / 2 + step * (i + 0.5), y - vert_gap, step)
        recurse(root, xcenter, vert_loc, width)
        return pos

    def get_params(self, deep=True):
        return {
            "min_samples_leaf": self.min_leaf,
            "pruning": self.pruning,
            "show_tables": self.show_tbl,
            "auto_visualize": self.auto_vis
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key == "min_samples_leaf":
                self.min_leaf = value
            elif key == "pruning":
                self.pruning = value
            elif key == "show_tables":
                self.show_tbl = value
            elif key == "auto_visualize":
                self.auto_vis = value
        return self