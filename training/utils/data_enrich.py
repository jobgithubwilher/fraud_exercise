import pandas as pd
import numpy as np

class FraudDetectionDataset:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the FraudDetectionDataset with a DataFrame.

        Args:
            dataframe (pd.DataFrame): The original dataset to be enriched.
        """
        self.df = dataframe.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])  # Ensure proper datetime format

    def apply_price_transformations(self):
        """
        Apply various transformations to the 'price' column, including:
        - Logarithmic transformation
        - Square root transformation
        - Reciprocal transformation
        - Standardization (Z-Score)
        - Min-Max Scaling
        """
        # Ensure no negative or zero values for logarithmic transformation
        self.df['price_log'] = np.log1p(self.df['price'])  # log1p ensures log(0) = 0

        # Apply square root transformation
        self.df['price_sqrt'] = np.sqrt(self.df['price'])

        # Apply reciprocal transformation (replace zeros with NaN to avoid division errors)
        self.df['price_reciprocal'] = 1 / self.df['price'].replace(0, np.nan)
    
    def calculate_time_features(self):
        """
        Extract timestamp-based features and drop the original timestamp column.
        Features include:
        - Day of the week
        - Whether the transaction occurred on a weekend
        - Hour of the day
        - Period of the day (morning, afternoon, evening, night)
        """
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['hour_of_day'] = self.df['timestamp'].dt.hour
        self.df['time_period'] = pd.cut(
            self.df['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        )
        # Convert time_period to a dummy variable
        self.df = pd.get_dummies(self.df, columns=['time_period'], prefix='period')
        # Drop the original timestamp column
        # self.df.drop(columns=['timestamp'], inplace=True)

    def calculate_session_features(self):
        """
        Calculate session-level features for each transaction, including:
        - Time since the first transaction in the session (using only previous transactions)
        - Total transactions so far (including the current one)
        - Average price so far (excluding current)
        - Total price so far (excluding current)
        """
        session_features = []
        
        for session_id, session_data in self.df.groupby('session_id'):
            session_data = session_data.sort_values('timestamp')  # Ensure chronological order
            
            # Initialize lists for the new features
            time_since_first = []
            total_transactions_so_far = []
            avg_price_so_far = []
            total_price_so_far = []
            
            # Iterate over each row in the session
            for idx, row in session_data.iterrows():
                # Get the transactions before the current one (excluding the current transaction)
                prev_transactions = session_data.loc[session_data.index < idx]
                
                # Calculate time_since_first (based on previous transactions)
                time_since_first.append((row['timestamp'] - session_data['timestamp'].min()).total_seconds())
                
                # Total transactions so far (count of previous + current)
                total_transactions_so_far.append(len(prev_transactions) + 1)
                
                # Calculate avg_price_so_far (excluding the current row)
                avg_price_so_far.append(prev_transactions['price'].mean() if not prev_transactions.empty else 0)
                
                # Calculate total_price_so_far (excluding the current row)
                total_price_so_far.append(prev_transactions['price'].sum())
            
            # Add the new features to the session data
            session_data['time_since_first'] = time_since_first
            session_data['total_transactions_so_far'] = total_transactions_so_far
            session_data['avg_price_so_far'] = avg_price_so_far
            session_data['total_price_so_far'] = total_price_so_far
            
            # Append the enriched session data to the list
            session_features.append(session_data)
        
        # Concatenate the session features back to the dataframe
        self.df = pd.concat(session_features).sort_index()


    def calculate_rolling_features(self):
        """
        Calculate rolling features based on the last 3 transactions in the session, including:
        - Rolling sum of prices
        - Rolling mean of prices
        - Rolling standard deviation of prices
        """
        rolling_features = []
        
        for session_id, session_data in self.df.groupby('session_id'):
            session_data = session_data.sort_values('timestamp')  # Ensure chronological order
            
            # Initialize lists for the rolling features
            rolling_sum_3 = []
            rolling_mean_3 = []
            rolling_std_3 = []
            
            # Iterate over each transaction in the session
            for idx, row in session_data.iterrows():
                # Get the previous transactions (excluding the current one)
                prev_transactions = session_data.loc[session_data.index < idx]
                
                # Get the last 3 previous transactions for rolling calculations
                prev_3 = prev_transactions.tail(3)
                
                # If there are less than 3 previous transactions, set to -1 (or another value)
                if len(prev_3) < 3:
                    rolling_sum_3.append(-1)
                    rolling_mean_3.append(-1)
                    rolling_std_3.append(-1)
                else:
                    # Calculate rolling sum of prices (last 3 previous transactions)
                    rolling_sum_3.append(prev_3['price'].sum())
                    
                    # Calculate rolling mean of prices (last 3 previous transactions)
                    rolling_mean_3.append(prev_3['price'].mean())
                    
                    # Calculate rolling standard deviation of prices (last 3 previous transactions)
                    rolling_std_3.append(prev_3['price'].std())
            
            # Add the new rolling features to the session data
            session_data['rolling_sum_3'] = rolling_sum_3
            session_data['rolling_mean_3'] = rolling_mean_3
            session_data['rolling_std_3'] = rolling_std_3
            
            # Append the enriched session data to the list
            rolling_features.append(session_data)
        
        # Concatenate the rolling features back to the dataframe
        self.df = pd.concat(rolling_features).sort_index()



    def encode_device(self):
        """
        Encode the 'device' column as a dummy variable for ML modeling.
        """
        self.df = pd.get_dummies(self.df, columns=['device'], prefix='device')

    def drop_unused_features(self):
        """
        Drop unused features that are not useful for ML modeling.
        Keeps the target column ('is_fraud').
        """
        self.df.drop(columns=['session_id'], inplace=True)
        self.df.drop(columns=['timestamp'], inplace=True)
        self.df.drop(columns=['device_ios'], inplace=True)
        self.df.drop(columns=['period_Evening'], inplace=True)

    def enrich_features(self):
        """
        Enrich the dataset with new features by sequentially calling all feature engineering methods.
        """
        self.calculate_time_features()
        self.calculate_session_features()
        self.calculate_rolling_features()
        self.encode_device()
        self.apply_price_transformations()
        self.drop_unused_features()

    def get_dataset(self):
        """
        Return the enriched dataset.

        Returns:
            pd.DataFrame: The enriched dataset with all additional features.
        """
        return self.df
