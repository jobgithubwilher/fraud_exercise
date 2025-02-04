import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import shap
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class FraudModelTrainer:
    """
    A class for training and evaluating fraud detection models, selecting the best model based on AUCPR,
    and optimizing thresholds for maximizing profit uplift.
    """

    def __init__(self, train_data, target_column='is_fraud', test_data=None, order_price_column='price'):
        self.train_data = train_data
        self.target_column = target_column
        self.test_data = test_data
        self.order_price_column = order_price_column
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        self.results = {}
        self.best_model = None
        self.best_threshold = None

    def save_best_model(self, model, model_name):
        if not os.path.exists('artifacts'):
            os.makedirs('artifacts')

        filename = os.path.join('artifacts', f"{model_name}_final_model.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved as {filename}")

    def preprocess_data(self):
        X = self.train_data.drop(columns=[self.target_column])
        y = self.train_data[self.target_column]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        self.X_train = X_train_balanced
        self.y_train = y_train_balanced
        self.X_val = X_val
        self.y_val = y_val

    def calculate_profit_uplift(self, y_true, y_scores, order_prices, threshold):
        y_pred = (y_scores >= threshold).astype(int)
        profit = 0

        for is_fraud, prediction, price in zip(y_true, y_pred, order_prices):
            if prediction == 0:
                if is_fraud == 0:
                    profit += 0.01 * price
                else:
                    profit -= price

        return profit

    def find_best_threshold(self, y_true, y_scores, order_prices):
        thresholds = np.linspace(0.1, 0.9, 9)
        best_threshold = 0.5
        best_profit = float('-inf')

        for threshold in thresholds:
            uplift = self.calculate_profit_uplift(y_true, y_scores, order_prices, threshold)
            if uplift > best_profit:
                best_profit = uplift
                best_threshold = threshold

        return best_threshold, best_profit

    def train_models(self):
        self.preprocess_data()

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(self.X_train, self.y_train)

            y_scores = model.predict_proba(self.X_val)[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_val, y_scores)
            aucpr = auc(recall, precision)

            best_threshold, best_profit = self.find_best_threshold(
                self.y_val, y_scores, self.train_data.loc[self.X_val.index, self.order_price_column]
            )

            self.results[model_name] = {
                "model": model,
                "aucpr": aucpr,
                "best_threshold": best_threshold,
                "profit_uplift": best_profit
            }

            print(f"{model_name} - AUCPR: {aucpr:.4f}, Best Threshold: {best_threshold:.2f}, Profit Uplift: ${best_profit:.2f}")

        self.best_model_name = max(self.results, key=lambda x: self.results[x]['aucpr'])
        self.best_model = self.results[self.best_model_name]["model"]
        self.best_threshold = self.results[self.best_model_name]["best_threshold"]

        self.save_best_model(self.best_model, self.best_model.__class__.__name__)

    def evaluate_on_test_data(self):
        if self.test_data is None:
            raise ValueError("Test dataset is not provided.")

        X_test = self.test_data.drop(columns=[self.target_column])
        y_test = self.test_data[self.target_column]
        order_prices = self.test_data[self.order_price_column]

        y_scores = self.best_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_scores >= self.best_threshold).astype(int)

        profit_uplift = self.calculate_profit_uplift(y_test, y_scores, order_prices, self.best_threshold)

        # Portfolio analysis
        accepted_orders = (y_test_pred == 0)
        rejected_orders = (y_test_pred == 1)

        accepted_count = accepted_orders.sum()
        rejected_count = rejected_orders.sum()

        accepted_volume = order_prices[accepted_orders].sum()
        rejected_volume = order_prices[rejected_orders].sum()

        print("Evaluation on Test Data:")
        print(f"Best Threshold: {self.best_threshold:.4f}")
        print(f"Profit Uplift: ${profit_uplift:,.2f}")
        print("\nPortfolio Analysis:")
        print(f"Total Accepted Orders: {accepted_count} (Volume: ${accepted_volume:,.2f})")
        print(f"Total Rejected Orders: {rejected_count} (Volume: ${rejected_volume:,.2f})")

    def global_explainability(self):
        print("\nGlobal Explainability (Permutation Feature Importance):")
        
        # Define a custom F1 scorer for permutation importance
        def f1_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return f1_score(y, y_pred)

        # Perform permutation importance
        perm_importance = permutation_importance(
            self.best_model, self.X_val, self.y_val, 
            scoring=f1_scorer, random_state=42
        )

        # Create a DataFrame to store permutation importance results
        perm_importance_df = pd.DataFrame({
            'Feature': self.X_val.columns,
            'Permutation Importance': perm_importance['importances_mean']
        }).sort_values(by='Permutation Importance', ascending=False)

        # Plot permutation importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Permutation Importance', y='Feature', data=perm_importance_df)
        plt.title('Permutation Feature Importance (F1 Score)', fontsize=16)
        plt.xlabel('Importance (Mean of Importances)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.show()

        print("\nPermutation Feature Importance Table:")
        print(perm_importance_df)

        print("\nGlobal Explainability (SHAP Feature Importance):")
        
        # Create SHAP explainer and calculate SHAP values
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(self.X_val)
        
        # Plot SHAP summary plot (red/blue dots)
        shap.summary_plot(shap_values, self.X_val, plot_type="dot", show=True)

        # Get mean absolute SHAP values for ranking features
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            'Feature': self.X_val.columns,
            'SHAP Importance': shap_importance
        }).sort_values(by='SHAP Importance', ascending=False)

        # Display SHAP feature importance table
        print("\nSHAP Feature Importance Table:")
        print(shap_importance_df)

    def explain(self):
        self.global_explainability()
