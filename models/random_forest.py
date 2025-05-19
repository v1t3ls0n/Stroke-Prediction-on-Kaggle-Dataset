from models.id3 import ID3DecisionTree
from models.c45 import C45Classifier
from models.c45_light import C45Classifier_Light
from models.cart import CARTDecisionTree
import numpy as np
import pandas as pd
from collections import Counter

class RandomForestCustom:
    def __init__(self, n_estimators=10, max_features='sqrt', tree_type='cart',
                 min_samples_leaf=2, bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.tree_type = tree_type.lower()
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.features_per_tree = []
        self.classes_ = None

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _get_base_tree(self):
        if self.tree_type == 'id3':
            return ID3DecisionTree()
        elif self.tree_type == 'c45':
            return C45Classifier()
        elif self.tree_type == 'c45_light':
            return C45Classifier_Light()
        elif self.tree_type == 'cart':
            return CARTDecisionTree(min_samples_leaf=self.min_samples_leaf)
        else:
            raise ValueError(f"Unsupported tree_type: {self.tree_type}")

    def _sample_features(self, n_total_features):
        if self.max_features == 'sqrt':
            return np.random.choice(n_total_features, int(np.sqrt(n_total_features)), replace=False)
        elif self.max_features == 'log2':
            return np.random.choice(n_total_features, int(np.log2(n_total_features)), replace=False)
        elif isinstance(self.max_features, int):
            return np.random.choice(n_total_features, self.max_features, replace=False)
        else:
            return np.arange(n_total_features)

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y).astype(int)
        self.classes_ = np.unique(y)
        self.trees = []
        self.features_per_tree = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(len(X), len(X), replace=True)
            else:
                indices = np.arange(len(X))

            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]

            feature_indices = self._sample_features(X.shape[1])
            selected_features = X.columns[feature_indices]
            X_selected = X_sample[selected_features]

            tree = self._get_base_tree()
            tree.fit(X_selected, y_sample)

            self.trees.append(tree)
            self.features_per_tree.append(selected_features)

    def predict(self, X):
        X = pd.DataFrame(X)
        predictions = []
        for tree, features in zip(self.trees, self.features_per_tree):
            preds = tree.predict(X[features])
            predictions.append(preds)

        predictions = np.array(predictions)
        maj_vote = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(maj_vote)

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        proba_sum = np.zeros((len(X), len(self.classes_)))
        for tree, features in zip(self.trees, self.features_per_tree):
            proba = tree.predict_proba(X[features])
            proba_sum += proba
        return proba_sum / self.n_estimators


    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "tree_type": self.tree_type,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self