"""
Cognitive Load Estimation Using Keystroke Dynamics

This script trains a Machine Learning model to predict cognitive load 
based on keystroke data. It demonstrates professional software engineering 
practices including Object-Oriented Programming (OOP), error handling, 
and model serialization for production readiness.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class CognitiveLoadModel:
    def __init__(self, data_path="keystroke_data.csv"):
        self.data_path = data_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_and_preprocess_data(self):
        """Loads data from CSV and applies preprocessing/scaling."""
        try:
            print(f"[INFO] Loading dataset from '{self.data_path}'...")
            data = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"[ERROR] Dataset '{self.data_path}' not found.")
            return None, None, None, None

        X = data.drop("load", axis=1)
        y = data["load"]
        self.feature_names = X.columns

        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # Feature Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train):
        """Trains the Random Forest model."""
        print("[INFO] Training Random Forest Classifier...")
        self.model.fit(X_train, y_train)
        print("[INFO] Model training complete.")

    def evaluate(self, X_test, y_test):
        """Evaluates model performance and generates classification metrics."""
        print("[INFO] Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n[RESULT] Model Accuracy: {accuracy * 100:.2f}%\n")
        
        print("[RESULT] Classification Report:")
        # Display using the actual class names (Low, Medium, High)
        target_names = [str(cls) for cls in self.label_encoder.classes_]
        try:
            print(classification_report(y_test, y_pred, target_names=target_names))
        except ValueError:
            # Fallback if classes in test set don't match all original classes (small dataset issue)
            print(classification_report(y_test, y_pred))
            
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_feature_importance()

    def _plot_confusion_matrix(self, y_test, y_pred):
        """Generates and saves a confusion matrix plot."""
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(cm, cmap='Blues')
        plt.title('Confusion Matrix', pad=20)
        fig.colorbar(cax)
        
        # Add labels based on encoder classes if length matches
        classes = self.label_encoder.classes_
        if len(classes) == cm.shape[0]:
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center', color='black')
            
        plt.tight_layout()
        plt.savefig("confusion_matrix_enhanced.png")
        print("[INFO] Saved confusion matrix plot to 'confusion_matrix_enhanced.png'")

    def _plot_feature_importance(self):
        """Generates and saves a feature importance bar chart."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(8, 5))
        plt.title("Feature Importance in Cognitive Load Estimation")
        plt.bar(range(len(importances)), importances[indices], align="center", color='skyblue', edgecolor='black')
        
        if self.feature_names is not None:
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
            
        plt.tight_layout()
        plt.savefig("feature_importance_enhanced.png")
        print("[INFO] Saved feature importance plot to 'feature_importance_enhanced.png'")

    def save_model(self, filename="cognitive_load_model.pkl"):
        """Serializes the trained model and preprocessors for deployment."""
        print(f"[INFO] Saving model, scaler, and encoder to '{filename}'...")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'encoder': self.label_encoder
        }, filename)
        print("[INFO] Export complete.")

    def predict_sample(self, sample_input):
        """Demonstrates inference on a new sample array."""
        sample_scaled = self.scaler.transform(sample_input)
        prediction = self.model.predict(sample_scaled)
        predicted_label = self.label_encoder.inverse_transform(prediction)[0]
        print(f"\n[PREDICTION] Input Features: {sample_input[0]}")
        print(f"[PREDICTION] Estimated Cognitive Load: >> {predicted_label} <<")

if __name__ == "__main__":
    # 1. Initialize Pipeline
    pipeline = CognitiveLoadModel("keystroke_data.csv")
    
    # 2. Load and Process Data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data()
    
    if X_train is not None:
        # 3. Train Model
        pipeline.train(X_train, y_train)
        
        # 4. Evaluate Model
        pipeline.evaluate(X_test, y_test)
        
        # 5. Export Model for Deployment
        pipeline.save_model()
        
        # 6. Predict on Mock New Data (Simulating a user typing)
        mock_data = np.array([[40, 150, 90, 450, 0.08, 5]])
        pipeline.predict_sample(mock_data)
