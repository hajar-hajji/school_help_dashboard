import torch
import torch.nn as nn
import torch.optim as optim

# Définir un MLP (Multi-Layer Perceptron) avec une couche de représentation
class MLPComplexity(nn.Module):
    def __init__(self, input_size):
        """
        Initialisation du modèle MLP.

        Args:
            input_size: La taille de l'entrée, correspondant au nombre de features.
        """
        super(MLPComplexity, self).__init__()

        # Première couche entièrement connectée (input_size -> 64 neurones)
        self.fc1 = nn.Linear(input_size, 64)

        # Deuxième couche entièrement connectée (64 -> 32 neurones), 
        # cette couche de représentation réduit les données à 32 dimensions
        self.fc2 = nn.Linear(64, 32)

        # Troisième couche entièrement connectée (32 -> 1 neurone), 
        # prédiction finale de la note (output)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Passer les données à travers les différentes couches du réseau.

        Args:
            x (Tensor): Le tenseur d'entrée de taille (batch_size, input_size).

        Returns:
            output (Tensor): La prédiction finale, càd la note (de taille (batch_size, 1)).
            representation (Tensor): Le vecteur de représentation intermédiaire (de taille (batch_size, 32)).
        """
        # Appliquer la fonction d'activation ReLU à la sortie de la première couche
        x = torch.relu(self.fc1(x))

        # Obtenir le vecteur de représentation après la deuxième couche (32 dimensions)
        representation = torch.relu(self.fc2(x))

        # Obtenir la prédiction finale après la troisième couche
        output = self.fc3(representation)

        # Retourner à la fois la sortie (output) et la représentation intermédiaire (representation)
        return output, representation

def train_mlp_with_early_stopping(model, X_train, y_train, X_val, y_val, epochs=10000, lr=10**-5, patience=1000):
    """
    Entraîne un modèle MLP avec l'early stopping
    
    Args:
        model: Le modèle MLP à entraîner.
        X_train: Les données d'entraînement (features).
        y_train: Les labels d'entraînement (valeurs cibles).
        X_val: Les données de validation (features).
        y_val: Les labels de validation (valeurs cibles).
        epochs: Le nombre maximum d'époques (itérations) pour l'entraînement. Valeur par défaut : 10000
        lr: Le taux d'apprentissage pour l'optimiseur Adam. Valeur par défaut : 10**-5
        patience: Nombre d'époques sans amélioration avant de déclencher l'early stopping. Valeur par défaut : 1000
    
    Returns:
        None: Le modèle est modifié en place avec les poids du meilleur modèle sauvegardé.
    """

    criterion = nn.MSELoss() # Fonction de perte
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Phase d'entraînement
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        
        # Forward pass
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Phase de validation
        model.eval()
        val_inputs = torch.tensor(X_val, dtype=torch.float32)
        val_labels = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            val_outputs, _ = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
        
        # Condition de l'early stopping
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        if patience_counter >= patience:
            model.load_state_dict(best_model)
            break