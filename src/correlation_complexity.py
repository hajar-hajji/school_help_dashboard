import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calculate_complexity_from_correlation(data, target_column):
    """
    Calcule la complexité d'accompagnement en utilisant les coefficients de corrélation des features
    avec la note finale. 
    
    Args:
    - data: dataframe initial contenant les données.
    - target_column: La colonne de la note finale, qui servira de cible pour le calcul de corrélation.
    
    Returns:
    - data: dataframe avec une nouvelle colonne 'Complexity' représentant la complexité normalisée.
    - correlations: Série contenant les coefficients de corrélation entre les features et la note finale.
    """
    
    # Séparation des features (X) et de la cible (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Calcul des coefficients de corrélation des features avec la note finale
    correlations = X.corrwith(y)
    
    # Calcul de la complexité en prenant en compte le signe des corrélations
    complexity = pd.Series(0, index=X.index)  # Initialise la série des complexités à zéro
    
    # Ajout ou soustraction de la contribution de chaque feature en fonction de la corrélation
    for feature in X.columns:
        if correlations[feature] >= 0:
            complexity += X[feature]*correlations[feature]
        else:
            complexity += (X[feature].max() - X[feature])*correlations[feature]
    
    # Normalisation entre 0 et 10
    scaler = MinMaxScaler(feature_range=(0,10))
    complexity = scaler.fit_transform(complexity.values.reshape(-1,1))
    
    data['Complexity'] = complexity
    
    return data, correlations