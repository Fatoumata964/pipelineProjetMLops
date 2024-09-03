from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Charger le modèle RandomForest
model = pickle.load(open("modelRF.pkl", "rb"))

def model_pred(features):
    # Créer un DataFrame à partir des caractéristiques fournies
    columns = [
        "credit_lines_outstanding",
        "loan_amt_outstanding",
        "total_debt_outstanding",
        "income",
        "years_employed",
        "fico_score"
    ]
    test_data = pd.DataFrame([features], columns=columns)
    
    # Prédire avec le modèle chargé
    prediction = model.predict(test_data)
    return int(prediction[0])

@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Récupérer les valeurs du formulaire
        credit_lines_outstanding = int(request.form["credit_lines_outstanding"])
        loan_amt_outstanding = float(request.form["loan_amt_outstanding"])
        total_debt_outstanding = float(request.form["total_debt_outstanding"])
        income = float(request.form["income"])
        years_employed = int(request.form["years_employed"])
        fico_score = int(request.form["fico_score"])

        # Prédire en utilisant les caractéristiques du formulaire
        prediction = model_pred([
            credit_lines_outstanding,
            loan_amt_outstanding,
            total_debt_outstanding,
            income,
            years_employed,
            fico_score
        ])

        # Afficher le résultat de la prédiction sur la page web
        if prediction == 1:
            return render_template(
                "index.html",
                prediction_text="Le client est susceptible de faire défaut."
            )
        else:
            return render_template(
                "index.html",
                prediction_text="Le client n'est pas susceptible de faire défaut."
            )

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
