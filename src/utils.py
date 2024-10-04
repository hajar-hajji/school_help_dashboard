from sklearn.model_selection import train_test_split
import torch

def split_data(data, target_column):
    """
    Divise le df en ensembles de données d'entraînement, de validation et de test.
    """

    X = data.drop(columns=[target_column]).values
    y = data[target_column].values
    
    # Diviser en train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_student_representation(model, student_data):
    """
    Obtient la représentation vectorielle d'un étudiant à partir du MLP entraîné.
    """
    
    # Convertir en tenseur et passer par le modèle
    student_tensor = torch.tensor(student_data, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        _, representation = model(student_tensor)
    
    return representation.squeeze().numpy()