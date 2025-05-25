import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import shap
import lime
import lime.lime_tabular

# ──────────────────────────────────────────────────────────────────────────────
# 1. PRÉTRAITEMENT DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_for_classification(df):
    """
    Nettoie les données et encode les variables pour la classification.
    """
    df = df.dropna()
    df = df[df['gender'].isin(['Male', 'Female'])]
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['employment_status'] = df['employment_status'].astype('category').cat.codes
    df['work_environment'] = df['work_environment'].astype('category').cat.codes
    df['mental_health_history'] = df['mental_health_history'].map({'Yes': 1, 'No': 0})
    df['seeks_treatment'] = df['seeks_treatment'].map({'Yes': 1, 'No': 0})
    df['mental_health_risk'] = df['mental_health_risk'].map({'Low': 0, 'Medium': 1, 'High': 2})
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 2. ENTRAÎNEMENT DES MODÈLES
# ──────────────────────────────────────────────────────────────────────────────
def train_classification_models(df, feature_cols, target_col='mental_health_risk'):
    """
    Entraîne plusieurs modèles de classification et retourne les résultats.
    """
    X = df[feature_cols]
    y = df[target_col]

    # Séparation en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Liste des modèles
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),  # LIME nécessite predict_proba
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    results = {}

    # Entraînement et évaluation
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print(f"\n=== {name} ===")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        results[name] = {
            'model': model,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }

    return results, X_train_scaled, X_test_scaled, y_test

# ──────────────────────────────────────────────────────────────────────────────
# 3. ÉVALUATION DES MODÈLES
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_classification_models(results, y_test, X_test_scaled):
    """
    Affiche les scores et matrices de confusion pour tous les modèles.
    """
    for name, content in results.items():
        model = content['model']
        y_pred = model.predict(X_test_scaled)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matrice de confusion - {name}")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 4. EXPLICATION GLOBALE AVEC SHAP
# ──────────────────────────────────────────────────────────────────────────────
def explain_model_shap(model, X_train_scaled, feature_names, nb_features=10):
    """
    Affiche l'importance globale des variables avec SHAP (summary plot).
    """
    explainer = shap.Explainer(model, X_train_scaled, feature_names=feature_names)
    shap_values = explainer(X_train_scaled)
    shap.summary_plot(shap_values, X_train_scaled, feature_names=feature_names, max_display=nb_features)

# ──────────────────────────────────────────────────────────────────────────────
# 5. EXPLICATION LOCALE AVEC LIME
# ──────────────────────────────────────────────────────────────────────────────
def explain_instance_lime(model, X_train_scaled, X_test_scaled, feature_names, class_names, instance_index=0):
    """
    Affiche l'explication LIME pour une prédiction individuelle.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_scaled),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=X_test_scaled[instance_index],
        predict_fn=model.predict_proba
    )
    explanation.show_in_notebook(show_table=True)

# ──────────────────────────────────────────────────────────────────────────────
# 6. SHAP GLOBAL + LOCAL POUR TOUS LES MODÈLES
# ──────────────────────────────────────────────────────────────────────────────
def explain_all_models_shap(results, X_train_scaled, X_test_scaled, feature_names, class_names, instance_index=0):
    """
    Affiche SHAP (global + local) pour tous les modèles du dictionnaire `results`.
    """
    for name, content in results.items():
        model = content['model']
        print(f"\n=== SHAP (global & local) pour le modèle : {name} ===")
        try:
            explainer = shap.Explainer(model, X_train_scaled, feature_names=feature_names)
            shap_values = explainer(X_train_scaled)

            # Importance globale
            print("Importance globale (SHAP Summary Plot) :")
            shap.summary_plot(shap_values, X_train_scaled, feature_names=feature_names, show=True)

            # Explication locale
            print(f"Explication locale pour l'observation {instance_index} :")
            shap.plots.waterfall(shap_values[instance_index], max_display=10)

        except Exception as e:
            print(f"Impossible d'expliquer {name} avec SHAP : {e}")
