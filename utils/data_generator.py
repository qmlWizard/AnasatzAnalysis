import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_moons, make_swiss_roll, make_gaussian_quantiles, load_iris, fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class DataGenerator:

    def __init__(self, dataset_name=None, file_path=None, n_samples=1000, n_pca_features=None, scaler_min = -np.pi, scaler_max = np.pi):
        self.dataset_name = dataset_name
        self.file_path = file_path
        self.n_samples = n_samples
        self.n_pca_features = n_pca_features
        self.dmin, self.dmax = 0, 1
        self._scaler_min = scaler_min
        self._scaler_max = scaler_max

    def generate_dataset(self):
        if self.file_path:  # Load dataset from a file if file_path is provided
            return self.load_from_file()
        elif self.dataset_name == 'iris':
            iris = load_iris()
            X = iris.data
            y = iris.target + 1  # Shift labels to start from 1 (1, 2, 3 instead of 0, 1, 2)
        elif self.dataset_name == 'mnist_fashion':
            X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
            y = y.astype(int)

            # Filter for shirts (class 6) and pants (class 1)
            binary_classes = [1, 6]  # Pants: 1, Shirts: 6
            mask = np.isin(y, binary_classes)
            X = X[mask]
            y = y[mask]

            # Map labels: Pants -> 1, Shirts -> -1
            y = np.where(y == 1, 1, -1)
        
        elif self.dataset_name == 'pcb_defect':
            pass

        else:
            raise ValueError("Dataset not supported. Choose from  'iris', 'mnist_fashion'.")

        # Apply MinMax scaling to the range [0, π]
        scaler = MinMaxScaler(feature_range=(self._scaler_min, self._scaler_max))
        X_scaled = scaler.fit_transform(X)

        # Apply PCA if specified
        if self.n_pca_features:
            X_scaled = self.apply_pca(X_scaled)

        # Return the scaled data as a DataFrame and the labels as a Series
        return pd.DataFrame(X_scaled, columns=[f'Feature {i+1}' for i in range(X_scaled.shape[1])]), pd.Series(y, name='Label')

    def apply_pca(self, X):
        """Apply PCA to reduce features."""
        if not self.n_pca_features:
            raise ValueError("Number of PCA features not specified.")
        pca = PCA(n_components=self.n_pca_features)
        X_pca = pca.fit_transform(X)
        return X_pca

    def load_from_file(self):
        """Load a dataset from a file and return a merged pandas DataFrame and Series."""
        data = np.load(self.file_path, allow_pickle=True).item()
        x_train, x_test = data['x_train'], data['x_test']
        y_train, y_test = data['y_train'], data['y_test']

        # Apply Min-Max Scaling to the range [0, π]
        scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Merge train and test data
        X_scaled = np.vstack([x_train_scaled, x_test_scaled])
        y = np.hstack([y_train, y_test])

        # Apply PCA if specified
        if self.n_pca_features:
            X_scaled = self.apply_pca(X_scaled)

        # Return as pandas DataFrame and Series
        return (
            pd.DataFrame(X_scaled, columns=[f'Feature {i+1}' for i in range(X_scaled.shape[1])]),
            pd.Series(y, name='Label')
        )

    def plot_dataset(self, df_features, df_labels):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('seaborn-v0_8')
        ax.scatter(df_features.iloc[:, 0], df_features.iloc[:, 1], c=df_labels, cmap='viridis', s=10, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.dataset_name} Dataset', fontsize=14, fontweight='bold')
        plt.colorbar(ax.scatter(df_features.iloc[:, 0], df_features.iloc[:, 1], c=df_labels, cmap='viridis', s=10, edgecolor='black', alpha=0.8), label='Label')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()
        return fig
