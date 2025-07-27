from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# ✅ Load model from model folder
model = pickle.load(open('model/Regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Get all inputs from form
        features = [
            'GrLivArea', 'OverallQual', 'KitchenQual', 'GarageArea', 'GarageCars',
            'TotalBsmtSF', 'ExterQual', '1stFlrSF', 'BsmtQual', 'GarageFinish',
            'FullBath', 'TotRmsAbvGrd', 'Foundation_PConc', '2ndFlrSF', 'BsmtFinSF1',
            'Fireplaces', 'LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtFinType1'
        ]

        # Get values from form
        input_data = [float(request.form[feature]) for feature in features]
        final_features = np.array([input_data])

        # ✅ Predict
        prediction = model.predict(final_features)[0]
        predicted_price = round(prediction, 2)

        return f"<h2 style='text-align:center;'>🏠 Predicted House Price: ${predicted_price}</h2>"

    except Exception as e:
        return f"Prediction failed:<br><pre>{str(e)}</pre>"

if __name__ == "__main__":
    app.run(port=5001, debug=True)


