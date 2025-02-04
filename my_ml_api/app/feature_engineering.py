import pandas as pd


class FraudDetectionDataset:
    @staticmethod
    def transform(input_data):
        """
        Transforms raw input data.
        """
        # Extract timestamp components
        hour = input_data.timestamp.hour
        day_of_week = input_data.timestamp.weekday()  # 0 = Monday, 6 = Sunday
        is_weekend = 1 if day_of_week >= 5 else 0  # Sat & Sun are weekends
        period_Afternoon = 1 if 12 <= hour < 18 else 0  # Afternoon period

        # Simulated placeholder values (replace with actual logic)
        avg_price_so_far = input_data.price  # Adjust based on logic
        rolling_sum_3 = 0  # Placeholder, replace with actual rolling sum logic
        rolling_mean_3 = 0  # Placeholder, replace with rolling mean logic
        rolling_std_3 = 0  # Placeholder, replace with rolling std logic
        total_transactions_so_far = 0  # Placeholder, replace with logic
        total_price_so_far = input_data.price  # Adjust based on real calc
        time_since_first = 0  # Placeholder, replace with actual logic

        return {
            "avg_price_so_far": avg_price_so_far,
            "rolling_sum_3": rolling_sum_3,
            "hour_of_day": hour,
            "rolling_mean_3": rolling_mean_3,
            "rolling_std_3": rolling_std_3,
            "device_android": (
                1 if input_data.device.lower() == "android" else 0
            ),
            "total_transactions_so_far": total_transactions_so_far,
            "time_since_first": time_since_first,
            "total_price_so_far": total_price_so_far,
            "period_Afternoon": period_Afternoon,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "price": input_data.price,
        }

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the FraudDetectionDataset with a DataFrame.

        Args:
            dataframe (pd.DataFrame): The original dataset to be enriched.
        """
        self.df = dataframe.copy()
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

    def calculate_time_features(self):
        """
        Extract timestamp-based features.
        """
        self.df["day_of_week"] = self.df["timestamp"].dt.dayofweek
        self.df["is_weekend"] = self.df["day_of_week"].isin([5, 6]).astype(int)
        self.df["hour_of_day"] = self.df["timestamp"].dt.hour
        self.df["period_Afternoon"] = (
            (self.df["hour_of_day"] >= 12) & (self.df["hour_of_day"] < 18)
        ).astype(int)

    def calculate_session_features(self):
        """
        Calculate session-level features for each transaction.
        """
        session_features = []

        for session_id, session_data in self.df.groupby("session_id"):
            # Ensure chronological order
            session_data = session_data.sort_values("timestamp")

            # Initialize lists for the new features
            time_since_first = []
            total_transactions_so_far = []
            avg_price_so_far = []
            total_price_so_far = []

            for idx, row in session_data.iterrows():
                prev_transactions = session_data.loc[session_data.index < idx]

                time_since_first.append(
                    (
                        row["timestamp"] - session_data["timestamp"].min()
                    ).total_seconds()
                )

                total_transactions_so_far.append(len(prev_transactions) + 1)

                avg_price_so_far.append(
                    prev_transactions["price"].mean()
                    if not prev_transactions.empty
                    else 0
                )

                total_price_so_far.append(prev_transactions["price"].sum())

            session_data["time_since_first"] = time_since_first
            session_data["total_transactions_so_far"] = (
                total_transactions_so_far
            )
            session_data["avg_price_so_far"] = avg_price_so_far
            session_data["total_price_so_far"] = total_price_so_far

            session_features.append(session_data)

        self.df = pd.concat(session_features).sort_index()

    def calculate_rolling_features(self):
        """
        Calculate rolling features based on the last 3 transactions.
        """
        rolling_features = []

        for session_id, session_data in self.df.groupby("session_id"):
            session_data = session_data.sort_values("timestamp")

            rolling_sum_3 = []
            rolling_mean_3 = []
            rolling_std_3 = []

            for idx, row in session_data.iterrows():
                prev_3 = session_data.loc[session_data.index < idx].tail(3)

                if len(prev_3) < 3:
                    rolling_sum_3.append(-1)
                    rolling_mean_3.append(-1)
                    rolling_std_3.append(-1)
                else:
                    rolling_sum_3.append(prev_3["price"].sum())
                    rolling_mean_3.append(prev_3["price"].mean())
                    rolling_std_3.append(prev_3["price"].std())

            session_data["rolling_sum_3"] = rolling_sum_3
            session_data["rolling_mean_3"] = rolling_mean_3
            session_data["rolling_std_3"] = rolling_std_3

            rolling_features.append(session_data)

        self.df = pd.concat(rolling_features).sort_index()

    def encode_device(self):
        """
        One-hot encode 'device' column, keeping only 'device_android'.
        """
        self.df = pd.get_dummies(self.df, columns=["device"], prefix="device")
        if "device_android" not in self.df.columns:
            self.df["device_android"] = 0  # Ensure feature exists

    def enrich_features(self):
        """
        Enrich the dataset with required features.
        """
        self.calculate_time_features()
        self.calculate_session_features()
        self.calculate_rolling_features()
        self.encode_device()

    def get_dataset(self):
        """
        Return the processed dataset with required features.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        return self.df
