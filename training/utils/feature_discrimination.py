import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency, ks_2samp
from sklearn.metrics import roc_auc_score

class FeatureEvaluator:
    def __init__(self, df, target):
        """
        Initialize the FeatureEvaluator class.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            target (str): Target variable name.
        """
        self.df = df
        self.target = target

    def evaluate_continuous_numerical(self, feature):
        """
        Evaluate the discriminatory power of a continuous numerical feature.

        Args:
            feature (str): Feature name.
        """
        fraud = self.df[self.df[self.target] == True][feature]
        non_fraud = self.df[self.df[self.target] == False][feature]

        # Statistical test: t-test
        _, p_value = ttest_ind(fraud, non_fraud, nan_policy="omit")

        # Visualization
        plt.figure(figsize=(8, 4))
        sns.kdeplot(fraud, label="Fraud", fill=True, color="red")
        sns.kdeplot(non_fraud, label="Non-Fraud", fill=True, color="blue")
        plt.title(f"Distribution of {feature} by {self.target}")
        plt.legend()
        plt.show()

        has_power = "Yes" if p_value < 0.05 else "No"
        return {"p_value": p_value, "discriminatory_power": has_power}

    def evaluate_integer_numerical(self, feature):
        """
        Evaluate the discriminatory power of an integer numerical feature.

        Args:
            feature (str): Feature name.
        """
        fraud = self.df[self.df[self.target] == True][feature]
        non_fraud = self.df[self.df[self.target] == False][feature]

        # Statistical test: t-test
        _, p_value = ttest_ind(fraud, non_fraud, nan_policy="omit")

        has_power = "Yes" if p_value < 0.05 else "No"
        return {"p_value": p_value, "discriminatory_power": has_power}

    def evaluate_categorical(self, feature):
        """
        Evaluate the discriminatory power of a categorical feature.

        Args:
            feature (str): Feature name.
        """
        contingency_table = pd.crosstab(self.df[feature], self.df[self.target])

        # Chi-square test
        _, p_value, _, _ = chi2_contingency(contingency_table)

        has_power = "Yes" if p_value < 0.05 else "No"
        return {"p_value": p_value, "discriminatory_power": has_power}

    def evaluate_feature(self, feature):
        """
        Determine feature type and evaluate its discriminatory power.

        Args:
            feature (str): Feature name.
        """
        if pd.api.types.is_float_dtype(self.df[feature]):
            return self.evaluate_continuous_numerical(feature)
        elif pd.api.types.is_integer_dtype(self.df[feature]):
            return self.evaluate_integer_numerical(feature)
        elif pd.api.types.is_categorical_dtype(self.df[feature]) or self.df[feature].nunique() < 10:
            return self.evaluate_categorical(feature)
        else:
            raise ValueError(f"Unsupported feature type for {feature}")

    def evaluate_all(self):
        """
        Evaluate all features in the dataset.
        """
        results = {}
        for feature in self.df.columns:
            if feature != self.target:
                try:
                    results[feature] = self.evaluate_feature(feature)
                except Exception as e:
                    print(f"Could not evaluate feature {feature}: {e}")
        return results