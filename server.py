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


# from flask import Flask, request, render_template
# import numpy as np
# import pickle
# from supabase_utils import insert_prediction_record, fetch_all_predictions

# app = Flask(__name__)

# # Load trained model
# model = pickle.load(open('model/Regmodel.pkl', 'rb'))

# features = [
#     'GrLivArea', 'OverallQual', 'KitchenQual', 'GarageArea', 'GarageCars',
#     'TotalBsmtSF', 'ExterQual', '1stFlrSF', 'BsmtQual', 'GarageFinish',
#     'FullBath', 'TotRmsAbvGrd', 'Foundation_PConc', '2ndFlrSF', 'BsmtFinSF1',
#     'Fireplaces', 'LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtFinType1'
# ]

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         input_data = [float(request.form[feature]) for feature in features]
#         final_features = np.array([input_data])
#         prediction = model.predict(final_features)[0]
#         predicted_price = round(prediction, 2)

#         # Prepare data to insert into Supabase
#         data = {feature: float(request.form[feature]) for feature in features}
#         data['predicted_price'] = predicted_price
#         insert_prediction_record(data)

#         return render_template('predict.html', result=predicted_price)

#     except Exception as e:
#         return f"Error: <pre>{str(e)}</pre>"

# @app.route('/report')
# def report():
#     properties = fetch_all_predictions()
#     return render_template('report.html', properties=properties)

# if __name__ == "__main__":
#     app.run(debug=True, port=5001)


# from flask import Flask, request, render_template
# import numpy as np
# from supabase_utils import insert_prediction_record

# @app.route('/predict', methods=['POST']) # type: ignore
# def predict():
#     try:
#         features = [
#             'GrLivArea', 'OverallQual', 'KitchenQual', 'GarageArea', 'GarageCars',
#             'TotalBsmtSF', 'ExterQual', '1stFlrSF', 'BsmtQual', 'GarageFinish',
#             'FullBath', 'TotRmsAbvGrd', 'Foundation_PConc', '2ndFlrSF', 'BsmtFinSF1',
#             'Fireplaces', 'LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtFinType1'
#         ]

#         input_data = {feature: float(request.form[feature]) for feature in features}
#         final_features = np.array([list(input_data.values())])

#         # Predict
#         prediction = model.predict(final_features)[0] # type: ignore
#         predicted_price = round(prediction, 2)

#         # 🔁 Save to Supabase
#         input_data['predicted_price'] = predicted_price
#         response = insert_prediction_record(input_data)
#         print("✅ Supabase insert response:", response)

#         return render_template("predict.html", result=predicted_price)

#     except Exception as e:
#         return f"Prediction failed:<br><pre>{str(e)}</pre>"


# from flask import Flask, request, render_template
# import numpy as np
# import pickle
# from supabase_utils import insert_prediction_record
# import os

# # Load model from relative path
# model_path = os.path.abspath("model/Regmodel.pkl")

# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model not found at {model_path}")

# with open(model_path, "rb") as f:
#     model = pickle.load(f)

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('predict.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         location = request.form['location']
#         area = float(request.form['area'])
#         bhk = int(request.form['bhk'])
#         bathroom = int(request.form['bathroom'])

#         # Example feature vector: you may need to preprocess this based on your model
#         features = np.array([[area, bhk, bathroom]])  # adjust if you have more inputs

#         prediction = model.predict(features)[0]
#         predicted_price = round(prediction, 2)

#         # Save to Supabase
#         record = {
#             "location": location,
#             "area": area,
#             "bhk": bhk,
#             "bathroom": bathroom,
#             "predicted_price": predicted_price
#         }
#         insert_prediction_record(record)

#         return render_template('predict.html', prediction=predicted_price)
    
#     except Exception as e:
#         return f"Error: {str(e)}"

# if __name__ == "__main__":
#     app.run(debug=True, port=5001)


# from flask import Flask, request, render_template, session, redirect, url_for
# import numpy as np
# import pickle
# from supabase_utils import insert_prediction_record, login_user, signup_user
# import os
# from dotenv import load_dotenv


# app = Flask(__name__)

# load_dotenv()
# app.secret_key = os.getenv("FLASK_SECRET_KEY")


# # Load model
# model_path = os.path.abspath("model/Regmodel.pkl")
# with open(model_path, "rb") as f:
#     model = pickle.load(f)

# @app.route('/')
# def home():
#     if 'user' in session:
#         return render_template('index.html', user=session['user'])
#     return redirect('/login')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']
#         res = login_user(email, password)
#         if res.user:
#             session['user'] = res.user.email
#             return redirect('/')
#         return "Login failed"
#     return render_template('login.html')

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         name = request.form['name']
#         email = request.form['email']
#         password = request.form['password']
#         user, err = signup_user(email, password, name)
#         if err:
#             return f"Signup failed: {err}"
#         session['user'] = email
#         return redirect('/')
#     return render_template('signup.html')

# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect('/login')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect('/login')
    
#     try:
#         fields = [
#             'GrLivArea', 'OverallQual', 'KitchenQual', 'GarageArea', 'GarageCars',
#             'TotalBsmtSF', 'ExterQual', '1stFlrSF', 'BsmtQual', 'GarageFinish',
#             'FullBath', 'TotRmsAbvGrd', 'Foundation_PConc', '2ndFlrSF',
#             'BsmtFinSF1', 'Fireplaces', 'LotArea', 'LotFrontage',
#             'MasVnrArea', 'BsmtFinType1'
#         ]
#         values = [float(request.form[field]) for field in fields]
#         features = np.array([values])
#         prediction = model.predict(features)[0]
#         predicted_price = round(prediction, 2)

#         record = {field: values[i] for i, field in enumerate(fields)}
#         record['predicted_price'] = predicted_price
#         insert_prediction_record(record)

#         return render_template('index.html', prediction=predicted_price, user=session['user'])
#     except Exception as e:
#         return f"Prediction Error: {e}"

# if __name__ == "__main__":
#     app.run(debug=True, port=5001)


