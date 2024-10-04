import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """
    Charge les données à partir d'un fichier CSV.
    """
    data = pd.read_csv(filepath, sep=',')
    return data

def prepare_data(data, method='correlation'):
    """
    Prépare les données en fonction de la méthode choisie (correlation ou mlp)
    
    - Si la méthode est 'correlation', seules les colonnes numériques sont conservées
    - Si la méthode est 'MLP', les colonnes catégorielles sont encodées
    
    Args:
        data: Le dataframe brut.
        method (str): La méthode de préparation ('correlation' ou 'mlp').
    
    Returns:
        Le dataframe avec les données préparées.
    """
    
    # Suppression des colonnes non pertinentes
    data = data.drop(columns=['StudentID', 'FirstName', 'FamilyName'])
    
    if method == 'correlation':
        # Pour la corrélation, ne garder que les colonnes numériques
        data = data.select_dtypes(include=['int64'])
    
    elif method == 'mlp':
        # Pour le MLP, on encode les variables catégorielles
        categorical_columns = data.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()

        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])
    
    return data