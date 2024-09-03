from app import model_pred

# Définir de nouvelles données de test avec les colonnes spécifiques à votre modèle
new_data = {
    'credit_lines_outstanding': 5,
    'loan_amt_outstanding': 10000.0,
    'total_debt_outstanding': 25000.0,
    'income': 75000.0,
    'years_employed': 10,
    'fico_score': 720
}

def test_predict():
    # Appeler la fonction de prédiction avec les nouvelles données
    prediction = model_pred(new_data)
    
    # Assurez-vous de connaître l'attente correcte pour ces données (1 ou 0)
    assert prediction == 1, "La prédiction est incorrecte"
