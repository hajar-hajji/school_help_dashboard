import streamlit as st
import plotly.express as px
import torch
import joblib
import os
from src.data_preprocessing import load_data, prepare_data
from src.correlation_complexity import calculate_complexity_from_correlation
from src.mlp_complexity import MLPComplexity, train_mlp_with_early_stopping
from src.utils import split_data, get_student_representation

def run_dashboard():
    """
    Fonction principale pour lancer le dashboard Streamlit
    """
    # Charger les données
    data = load_data('data/data.csv')

    # Choisir la méthode de calcul de la complexité
    method = st.sidebar.selectbox(
        'Choisir la méthode de calcul de la complexité',
        ['Correlation', 'MLP'])

    # Préparer les données selon la méthode choisie
    data_prepared = prepare_data(data, method=method.lower())

    if method == 'Correlation':
        # Calculer la complexité avec la méthode de corrélation
        data_with_complexity, _ = calculate_complexity_from_correlation(data_prepared, target_column='FinalGrade')
    
    elif method == 'MLP':
        model_path = 'models/mlp_model.pkl'
        
        # Vérifier si le modèle existe
        if not os.path.exists(model_path):
            st.write("Modèle MLP non trouvé. Entraînement du modèle...")
            
            # Préparer les données pour le modèle MLP
            data_mlp = prepare_data(data, method='mlp')
            X_train, X_val, _, y_train, y_val, _ = split_data(data_mlp, target_column='FinalGrade')
            
            model = MLPComplexity(input_size=X_train.shape[1])
            
            # Entraîner le modèle
            train_mlp_with_early_stopping(model, X_train, y_train, X_val, y_val, epochs=10000, lr=10**-4, patience=1000)
            
            # Créer le dossier 'models' s'il n'existe pas encore
            os.makedirs('models', exist_ok=True)

            # Sauvegarder le modèle entraîné
            joblib.dump(model, model_path)
            st.write("Modèle MLP entraîné et sauvegardé !")

        # Charger le modèle MLP pré-entraîné
        model = joblib.load(model_path) 
        
        # Séparer les données
        X_full = data_prepared.drop(columns=['FinalGrade']).values
        y_full = data_prepared['FinalGrade'].values
        
        # Sélectionner l'étudiant ayant eu 20 comme référence
        student_index = (y_full == 20).argmax()  # Trouver l'indice de l'étudiant avec 20
        reference_student = X_full[student_index]
        
        # Obtenir la représentation vectorielle de l'étudiant ayant eu 20
        reference_representation = get_student_representation(model, reference_student)
        
        # Calculer la complexité pour tous les élèves dans le dataset complet
        complexities = []
        for student_data in X_full:
            representation = get_student_representation(model, student_data)
            complexity = torch.norm(torch.tensor(representation) - torch.tensor(reference_representation)).item()
            complexities.append(complexity)
    
        data_with_complexity = data_prepared.copy()
        data_with_complexity['Complexity'] = complexities

    data['Complexity'] = data_with_complexity['Complexity']
    # Filtres par sexe
    sex_options = data['sex'].unique()
    sex = st.sidebar.multiselect('Sexe', sex_options, default=sex_options)

    # Filtres par âge
    age_min = int(data['age'].min())
    age_max = int(data['age'].max())
    age = st.sidebar.slider('Âge', age_min, age_max, (age_min, age_max))

    # Filtres par absences
    absences_max = int(data['absences'].max())
    absences = st.sidebar.slider('Nombre d\'absences', 0, absences_max, (0, absences_max))

    # Appliquer les filtres
    filtered_data = data[
        (data['sex'].isin(sex)) &
        (data['age'] >= age[0]) &
        (data['age'] <= age[1]) &
        (data['absences'] <= absences[1])]

    # Créer un scatter plot des élèves filtrés
    fig_filtered = px.scatter(
        filtered_data,
        x='FinalGrade',
        y='Complexity',
        hover_data=['sex', 'age', 'absences', 'Dalc', 'Walc', 'studytime'],
        labels={'FinalGrade': 'Note Finale', 'Complexity': 'Complexité d\'accompagnement'},
        title='Priorisation des élèves pour l\'accompagnement')
    
    # Inverser l'axe des x
    fig_filtered.update_layout(xaxis=dict(autorange='reversed'))

    # Afficher le scatter plot
    st.plotly_chart(fig_filtered)

if __name__ == '__main__':
    # Lancer le dashboard
    run_dashboard()