import pandas as pd

class TimeSeriesSplitter:
    def __init__(self, df, train_size=0.8):
        """
        Initialize the TimeSeriesSplitter class.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            train_size (float): Proportion of the dataset to include in the training set (default is 0.8).
        """
        self.df = df
        self.train_size = train_size
        self.train_set = None
        self.test_set = None

    def split(self):
        """
        Split the dataset into training and test sets based on the specified train_size.
        """
        split_index = int(len(self.df) * self.train_size)
        self.train_set = self.df.iloc[:split_index]
        self.test_set = self.df.iloc[split_index:]

    def get_train_test_sets(self):
        """
        Get the training and test sets after the split.

        Returns:
            tuple: A tuple containing the training and test DataFrames.
        """
        if self.train_set is None or self.test_set is None:
            self.split()
        return self.train_set, self.test_set
