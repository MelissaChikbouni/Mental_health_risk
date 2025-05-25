# Importation des bibliothèques nécessaires
import pandas as pd  # pour la manipulation de données tabulaires
from sklearn.preprocessing import StandardScaler  # pour normaliser les données
from sklearn.decomposition import PCA  # pour la réduction de dimension
from sklearn.cluster import KMeans, DBSCAN  # pour les algorithmes de clustering
from sklearn.ensemble import IsolationForest  # pour la détection d’anomalies
from sklearn.neighbors import LocalOutlierFactor  # autre méthode de détection d’anomalies
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # scores d’évaluation de clustering
import matplotlib.pyplot as plt  # pour la visualisation
import seaborn as sns  
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt





# Fonction pour charger et préparer les données

def visualiser_donnees(df):
    """Affiche des visualisations groupées et compactes du dataset santé mentale"""
    sns.set(style="whitegrid")
    plt.figure(figsize=(18, 12))

    # --- 1. Distribution de l'âge ---
    plt.subplot(2, 3, 1)
    sns.histplot(df["age"], bins=20, kde=True, color='skyblue')
    plt.title("Âge")

    # --- 2. Répartition par genre ---
    plt.subplot(2, 3, 2)
    sns.countplot(x="gender", data=df, palette="Set2")
    plt.title("Genre")

    # --- 3. Stress vs antécédents santé mentale ---
    plt.subplot(2, 3, 3)
    sns.boxplot(x="mental_health_history", y="stress_level", data=df, palette="Set3")
    plt.title("Stress selon historique santé mentale")

    # --- 4. Activité physique par genre ---
    plt.subplot(2, 3, 4)
    sns.violinplot(x="gender", y="physical_activity_days", data=df, palette="pastel")
    plt.title("Activité physique / genre")
    # --- 5. Anxiété par statut professionnel ---
    plt.subplot(2, 3, 5)

    sns.boxplot(x="employment_status", y="anxiety_score", data=df, palette="coolwarm")
    plt.xticks(rotation=45)
    plt.title("Anxiété selon statut professionnel")
    plt.show()


    # --- 6. Matrice de corrélation ---
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.show()

def preprocess_data(df):    
    df = df.dropna()  # supprimer les lignes avec des valeurs manquantes

    # Filtrer pour ne garder que les genres homme/femme (exclusion des autres pour simplifier)
    df = df[df['gender'].isin(['Male', 'Female'])]  
    
    # Encodage binaire du genre : 0 pour homme, 1 pour femme
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    # Encodage des variables catégorielles en codes numériques
    df['employment_status'] = df['employment_status'].astype('category').cat.codes
    df['work_environment'] = df['work_environment'].astype('category').cat.codes

    # Encodage binaire des variables booléennes
    df['mental_health_history'] = df['mental_health_history'].map({'Yes': 1, 'No': 0})
    df['seeks_treatment'] = df['seeks_treatment'].map({'Yes': 1, 'No': 0})

    return df  # retourne le DataFrame préparé

# Fonction pour normaliser les données (centrer-réduire)
def scale_features(df, feature_cols):
    scaler = StandardScaler()  # initialise le scaler
    return scaler.fit_transform(df[feature_cols])  # retourne les données normalisées


# Réduction de dimension avec l'ACP (2 composantes principales)
def pca_reduction(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)  # applique la PCA
    return X_pca  # retourne les données projetées dans l’espace réduit

# Application du clustering K-Means
def apply_kmeans(X_scaled, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)  # prédit les clusters
    return labels, model  # retourne les étiquettes et le modèle










#------------Annnuler-----------------------
# Application de DBSCAN (clustering basé sur la densité)
def apply_dbscan(X_scaled, eps=1.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)  # retourne les labels (-1 = outliers)
    return labels

# Détection d’anomalies avec Isolation Forest
def detect_outliers_isolation_forest(X_scaled):
    iso = IsolationForest(contamination=0.1, random_state=42)  # 10% supposés anormaux
    return iso.fit_predict(X_scaled)  # -1 = anomalie, 1 = normal

# Détection d’anomalies avec LOF (Local Outlier Factor)
def detect_outliers_lof(X_scaled):
    lof = LocalOutlierFactor()
    return lof.fit_predict(X_scaled)  # -1 = anomalie, 1 = normal
#---------------------------------------------


# Calcul de l'inertie pour différents k (méthode du coude)
def elbow_method(X_scaled, max_k=10):
    inertias = []
    for k in range(2, max_k+1):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X_scaled)
        inertias.append(model.inertia_)  # inertie intra-cluster
    return inertias  # retourne la liste des inerties

# Visualisation du coude pour déterminer le meilleur k
def plot_elbow(inertias):
    plt.plot(range(2, len(inertias)+2), inertias, marker='o')
    plt.title("Méthode du coude")  # titre du graphe
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie intra-cluster")
    plt.grid(True)
    plt.show()  # affiche le graphe


# Entraînement de plusieurs modèles KMeans avec évaluation des scores
def entrainer_kmeans_multiples(X_scaled, k_range=range(2, 11)):
    resultats = []
    for k in k_range:
        labels, model = apply_kmeans(X_scaled, n_clusters=k)
        silhouette, calinski, davies = evaluate_clustering(X_scaled, labels)
        resultats.append({
            'k': k,
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies,
            'labels': labels,
            'model': model
        })
    return resultats

# Visualisation comparée des scores pour différents K
def visualiser_scores(resultats):
    import matplotlib.pyplot as plt

    ks = [r['k'] for r in resultats]
    silhouettes = [r['silhouette'] for r in resultats]
    calinskis = [r['calinski'] for r in resultats]
    db_scores = [r['davies'] for r in resultats]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].plot(ks, silhouettes, marker='o', label="Silhouette")
    ax[0].set_title("Silhouette Score")
    ax[0].set_xlabel("k")

    ax[1].plot(ks, calinskis, marker='o', label="Calinski-Harabasz", color="green")
    ax[1].set_title("Calinski-Harabasz Score")
    ax[1].set_xlabel("k")

    ax[2].plot(ks, db_scores, marker='o', label="Davies-Bouldin", color="red")
    ax[2].set_title("Davies-Bouldin Score")
    ax[2].set_xlabel("k")

    for ax_ in ax:
        ax_.grid(True)

    plt.suptitle("Comparaison des métriques de clustering")
    plt.tight_layout()
    plt.show()


# Identification des features les plus importantes dans chaque cluster
def top_features_par_cluster(X_scaled, feature_names, labels, top_n=5):


    df_features = pd.DataFrame(X_scaled, columns=feature_names)
    df_features['cluster'] = labels
    top_features = {}

    for cluster in sorted(df_features['cluster'].unique()):
        cluster_data = df_features[df_features['cluster'] == cluster].drop(columns=['cluster'])
        mean_values = cluster_data.mean()
        top = mean_values.abs().sort_values(ascending=False).head(top_n)
        top_features[cluster] = list(top.index)

    return top_features



# Évaluation d’un clustering à l’aide de plusieurs métriques
def evaluate_clustering(X_scaled, labels):
    # Si plusieurs clusters identifiés et pas de bruit (-1), on évalue
    if len(set(labels)) > 1 and -1 not in set(labels):
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
        return silhouette, calinski, davies
    return None, None, None  # si évaluation non applicable
def visualiser_clusters_pca(resultats, X_scaled):
    """
    Affiche les visualisations PCA des clusters (réduction à 2D) pour chaque KMeans entraîné avec k spécifique.
    Affiche 3 graphiques maximum par ligne.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    # Réduction en 2D avec PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    n = len(resultats)
    cols = 3  # max 3 graphiques par ligne
    rows = math.ceil(n / cols)

    plt.figure(figsize=(6 * cols, 5 * rows))

    for i, r in enumerate(resultats):
        labels = r['labels']
        k = r['k']

        plt.subplot(rows, cols, i + 1)
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', s=60, edgecolor='k')
        plt.title(f"Clusters PCA (k={k})")
        plt.xlabel("Composante 1")
        plt.ylabel("Composante 2")
        plt.xticks([])
        plt.yticks([])
        plt.legend(title="Cluster", loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyse_risque_par_cluster(df, labels, afficher_heatmap=True):
    """
    Associe les clusters aux individus, analyse la répartition des niveaux de risque
    et affiche une heatmap si demandé.

    Paramètres :
        df (DataFrame) : le DataFrame d'origine (doit contenir 'mental_health_risk')
        labels (array-like) : les labels de cluster obtenus via KMeans
        afficher_heatmap (bool) : si True, affiche une heatmap

    Retourne :
        DataFrame avec la proportion des niveaux de risque par cluster
    """
    df_temp = df.copy()
    df_temp['cluster'] = labels

    # Calcul des proportions de chaque niveau de risque dans chaque cluster
    tableau_risque = df_temp.groupby('cluster')['mental_health_risk'].value_counts(normalize=True).unstack().fillna(0)

    if afficher_heatmap:
        plt.figure(figsize=(8, 5))
        sns.heatmap(tableau_risque, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title("Répartition des niveaux de risque par cluster")
        plt.xlabel("Niveau de risque")
        plt.ylabel("Cluster")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    return tableau_risque


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyser_clusters_complet(df, labels, X_scaled, feature_names, afficher_heatmap=True, top_n=5):
    """
    Analyse complète des clusters : répartition des niveaux de risque et top features par cluster.

    Paramètres :
        df (DataFrame) : le jeu de données d'origine (doit contenir 'mental_health_risk')
        labels (array-like) : les labels de cluster obtenus par KMeans
        X_scaled (ndarray) : les données normalisées (résultat du StandardScaler)
        feature_names (list) : noms des colonnes utilisées pour X_scaled
        afficher_heatmap (bool) : True pour afficher une heatmap
        top_n (int) : nombre de variables les plus représentatives à afficher par cluster

    Retourne :
        - tableau_risque : DataFrame avec les proportions de niveaux de risque par cluster
        - top_features : dictionnaire des variables principales par cluster
    """

    # Ajout des clusters dans le DataFrame original (copie pour ne pas modifier df)
    df_temp = df.copy()
    df_temp['cluster'] = labels

    # --- 1. Répartition des niveaux de risque par cluster ---
    tableau_risque = df_temp.groupby('cluster')['mental_health_risk'].value_counts(normalize=True).unstack().fillna(0)

    if afficher_heatmap:
        plt.figure(figsize=(8, 5))
        sns.heatmap(tableau_risque, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title("Répartition des niveaux de risque par cluster")
        plt.xlabel("Niveau de risque")
        plt.ylabel("Cluster")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # --- 2. Identification des top features par cluster ---
    df_features = pd.DataFrame(X_scaled, columns=feature_names)
    df_features['cluster'] = labels
    top_features = {}

    for cluster in sorted(df_features['cluster'].unique()):
        # Moyenne absolue des features pour les individus d’un cluster
        cluster_data = df_features[df_features['cluster'] == cluster].drop(columns=['cluster'])
        mean_values = cluster_data.mean()
        top = mean_values.abs().sort_values(ascending=False).head(top_n)
        top_features[cluster] = list(top.index)

    return tableau_risque, top_features

