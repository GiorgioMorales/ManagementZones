import os
import numpy as np
import pandas as pd


class DataNormalization:
    """Normalize different types of input covariates"""
    def __init__(self, variables: list, save_path: str = None, schema: pd.DataFrame = None):
        """
        :param variables: List of variables names (includes the response variable name; e.g., 'yld')
        :param save_path: Path where the modified schema will be saved locally
        :param schema: If the provided data schema is None, the base schema will be loaded.
        """
        self.variables = variables
        self.save_path = save_path
        self.df = schema
        self.df_complete = None

    def load_base_schema(self):
        """Retrieve information from the base schema"""
        # Load base schema as df
        try:
            data = pd.read_csv('src/OFPEAI/Predictor/data_schema/base_schema.csv')
            df = pd.DataFrame(data)
        except FileNotFoundError:
            data = pd.read_csv(os.getcwd() + '/base_schema.csv')
            df = pd.DataFrame(data)
        return df, df[df['variable'].isin(self.variables)]  # Complete df and df with discarded rows

    def calculate_stats(self, X: np.ndarray, Y: np.ndarray):
        """Calculate statistics of the dataset and update data schema
        :param X: Input data. If None, this class can only be used to reverse normalization.
        :param Y: Target data. If None, this class can only be used to reverse normalization.
        """
        for n in range(len(self.df)):
            if n == 0:  # First row corresponds to the response variable
                data = Y
            else:
                # Find which index of X corresponds to the current row
                ind = np.where(np.array(self.variables) == self.df.iloc[n]['variable'])[0] - 1
                data = X[:, :, ind, :, :]
            if data is not None:
                self.df.iloc[n, self.df.columns.get_loc('mean')] = np.mean(data)
                self.df.iloc[n, self.df.columns.get_loc('std')] = np.std(data)
                if np.isnan(self.df.iloc[n]['min']):  # If the minimum value was not specified, calculate it
                    self.df.iloc[n, self.df.columns.get_loc('min')] = np.min(data)
                if np.isnan(self.df.iloc[n]['max']):  # If the minimum value was not specified, calculate it
                    self.df.iloc[n, self.df.columns.get_loc('max')] = np.max(data)

    def min_max(self, X: np.ndarray, n: int, top: float = 1):
        """Perform min_max normalization
        :param X: Data to be normalized corresponding to one variable
        :param n: Index of the variable in the data schema
        :param top: Maximum value the data will take after normalization (i.e., data will be scaled between 0 and 'top')
        """
        minV, maxV = self.df.iloc[n]['min'], self.df.iloc[n]['max']
        return (X - minV) / (maxV - minV) * top

    def reverse_min_max(self, X: np.ndarray, n: int, top: float = 1):
        """Reverse min_max normalization
        :param X: Data to be processed corresponding to one variable
        :param n: Index of the variable in the data schema
        :param top: Maximum value (i.e., data is scaled between 0 and 'top')
        """
        minV, maxV = self.df.iloc[n]['min'], self.df.iloc[n]['max']
        return (X * (maxV - minV) / top) + minV

    def z_score(self, X: np.ndarray, n: int):
        """Perform z-score normalization
        :param X: Data to be normalized corresponding to one variable
        :param n: Index of the variable in the data schema
        """
        meanV, stdV = self.df.iloc[n]['mean'], self.df.iloc[n]['std']
        return (X - meanV) / stdV

    def reverse_z_score(self, X: np.ndarray, n: int):
        """Reverse z-score normalization
        :param X: Data to be processed corresponding to one variable
        :param n: Index of the variable in the data schema
        """
        meanV, stdV = self.df.iloc[n]['mean'], self.df.iloc[n]['std']
        return (X * stdV) + meanV

    def normalization(self, X: np.ndarray, Y):
        """Perform normalization. The type of normalization depends on each variable
        :param X: Input data. If None, this class can only be used to reverse normalization.
        :param Y: Target data. If None, this class can only be used to reverse normalization.
        """
        # Load data schema as a dataframe
        if self.df is None:
            self.df_complete, self.df = self.load_base_schema()
            # Compute statistics and update schema
            self.calculate_stats(X, Y)
            # Save updated schema as CSV
            self.df_complete[self.df_complete['variable'].isin(self.variables)] = self.df
            if self.save_path is not None:
                self.df_complete.to_csv(self.save_path)

        # First row corresponds to the response variable. Use min-max between 0 and 10
        if Y is not None:  # During testing, only covariates X are normalized
            Y = self.min_max(X=Y, n=0, top=10)
        # Covariate normalization
        for n in range(1, len(self.df)):
            # Find which index of X corresponds to the current row
            ind = np.where(np.array(self.variables) == self.df.iloc[n]['variable'])[0] - 1
            if self.df.iloc[n]['normalization'] == 'min-max':  # Min-max normalization
                X[:, :, ind, :, :] = self.min_max(X=X[:, :, ind, :, :], n=n)
            else:  # Z-score normalization
                X[:, :, ind, :, :] = self.z_score(X=X[:, :, ind, :, :], n=n)
        return X, Y

    def reverse_normalization(self, X, Y):
        """Reverse normalization. The type of normalization depends on each variable
        :param X: Input data that will be reverted
        :param Y: Target data that will be reverted
        """
        # First row corresponds to the response variable. Use min-max between 0 and 10
        Y = self.reverse_min_max(X=Y, n=0, top=10)
        # Covariate normalization
        if X is not None:  # During training, only the targets will be reverted to evaluate performance
            for n in range(1, len(self.df)):
                if self.df.iloc[n]['normalization'] == 'min-max':  # Min-max normalization
                    X[:, :, n - 1, :, :] = self.reverse_min_max(X=X[:, :, n - 1, :, :], n=n)
                else:  # Z-score normalization
                    X[:, :, n - 1, :, :] = self.reverse_z_score(X=X[:, :, n - 1, :, :], n=n)
        return X, Y


if __name__ == '__main__':
    x = np.zeros((1, 1, 9, 5, 5))
    y = np.zeros((1, 1, 5, 5))
    data_norm = DataNormalization(['yld', 'aa_n', 'slope', 'elev', 'tpi', 'aspect_rad', 'prec_cy_g', 'vv_cy_f',
                                   'vh_cy_f', 'texture0cm'])
    data_norm.normalization(x, y)
