import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from boruta import BorutaPy

class FeatureSelector:
    def __init__(self, df, target):
        """
        Initialize the FeatureSelector class.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            target (str): Target variable name.
        """
        self.df = df
        self.target = target
        self.features = [col for col in df.columns if col != target]
        self.results = {}

    def boruta_selection(self):
        """
        Perform feature selection using the Boruta algorithm.
        """
        X = self.df[self.features].values
        y = self.df[self.target].values
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        boruta = BorutaPy(rf, n_estimators='auto', random_state=42)
        boruta.fit(X, y)
        selected_features = [feature for feature, support in zip(self.features, boruta.support_) if support]
        self.results['Boruta'] = selected_features

    def permutation_selection(self):
        """
        Perform feature selection using permutation importance.
        """
        X = self.df[self.features]
        y = self.df[self.target]
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        perm_importance = permutation_importance(rf, X, y, scoring='accuracy', random_state=42)
        threshold = np.percentile(perm_importance.importances_mean, 75)
        selected_features = [feature for feature, importance in zip(self.features, perm_importance.importances_mean) if importance >= threshold]
        self.results['Permutation'] = selected_features

    def select_k_best(self, k=10):
        """
        Perform feature selection using SelectKBest with ANOVA F-value.

        Args:
            k (int): Number of top features to select.
        """
        X = self.df[self.features]
        y = self.df[self.target]
        skb = SelectKBest(score_func=f_classif, k=k)
        skb.fit(X, y)
        selected_features = [feature for feature, support in zip(self.features, skb.get_support()) if support]
        self.results['SelectKBest'] = selected_features

    def rfe_selection(self, n_features_to_select=10):
        """
        Perform feature selection using Recursive Feature Elimination (RFE).

        Args:
            n_features_to_select (int): Number of features to select.
        """
        X = self.df[self.features]
        y = self.df[self.target]
        lr = LogisticRegression(max_iter=1000, random_state=42)
        rfe = RFE(lr, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        selected_features = [feature for feature, support in zip(self.features, rfe.support_) if support]
        self.results['RFE'] = selected_features

    def voting_system(self):
        """
        Aggregate results from all methods to select the best features.

        If a feature is selected by even one method, include it in the final list.
        """
        all_features = [feature for method_features in self.results.values() for feature in method_features]
        selected_features = list(set(all_features))  # Get unique features suggested by any method
        self.results['Voting'] = selected_features

    def select_features(self):
        """
        Perform all feature selection methods and apply voting.
        """
        # self.boruta_selection()
        self.permutation_selection()
        print("Permutation Selection Finished")
        self.select_k_best()
        print("Select K Bests Finished")
        #self.rfe_selection()
        self.voting_system()
        return self.results
