# import os
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Load dataset
# df = pd.read_csv('../data/train.csv')

# # Add derived column: 1 if Foundation is PConc else 0
# df['Foundation_PConc'] = df['Foundation'].apply(lambda x: 1 if x == 'PConc' else 0)

# # Define features based on HTML form inputs
# features = [
#     'GrLivArea', 'OverallQual', 'KitchenQual', 'GarageArea', 'GarageCars',
#     'TotalBsmtSF', 'ExterQual', '1stFlrSF', 'BsmtQual', 'GarageFinish',
#     'FullBath', 'TotRmsAbvGrd', 'Foundation_PConc', '2ndFlrSF', 'BsmtFinSF1',
#     'Fireplaces', 'LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtFinType1'
# ]

# # Keep only relevant features + target
# df = df[features + ['SalePrice']]

# # Handle categorical mappings
# qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0}
# finish_mapping = {'Fin': 3, 'RFn': 2, 'Unf': 1, np.nan: 0}
# bsmtfin_mapping = {
#     'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, np.nan: 0
# }

# df['ExterQual'] = df['ExterQual'].map(qual_mapping)
# df['KitchenQual'] = df['KitchenQual'].map(qual_mapping)
# df['BsmtQual'] = df['BsmtQual'].map(qual_mapping)
# df['GarageFinish'] = df['GarageFinish'].map(finish_mapping)
# df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmtfin_mapping)

# # Fill remaining NaNs
# df.fillna(0, inplace=True)

# # Train-test split
# X = df[features]
# y = df['SalePrice']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)

# # ✅ Ensure output directory exists
# output_dir = "../model"
# os.makedirs(output_dir, exist_ok=True)

# # ✅ Save the trained model
# model_path = os.path.join(output_dir, "Regmodel.pkl")
# with open(model_path, "wb") as f:
#     pickle.dump(model, f)

# print(f"✅ Model saved successfully at {model_path}")


# train_model.py (refactored)
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('./data/train.csv')

# Derived column
df['Foundation_PConc'] = df['Foundation'].apply(lambda x: 1 if x == 'PConc' else 0)

# Feature selection
features = [
    'GrLivArea', 'OverallQual', 'KitchenQual', 'GarageArea', 'GarageCars',
    'TotalBsmtSF', 'ExterQual', '1stFlrSF', 'BsmtQual', 'GarageFinish',
    'FullBath', 'TotRmsAbvGrd', 'Foundation_PConc', '2ndFlrSF', 'BsmtFinSF1',
    'Fireplaces', 'LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtFinType1'
]
df = df[features + ['SalePrice']]

# Categorical mapping
qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0}
finish_mapping = {'Fin': 3, 'RFn': 2, 'Unf': 1, np.nan: 0}
bsmtfin_mapping = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, np.nan: 0}

df['ExterQual'] = df['ExterQual'].map(qual_mapping)
df['KitchenQual'] = df['KitchenQual'].map(qual_mapping)
df['BsmtQual'] = df['BsmtQual'].map(qual_mapping)
df['GarageFinish'] = df['GarageFinish'].map(finish_mapping)
df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmtfin_mapping)

# Fill missing values
df.fillna(0, inplace=True)

# Split data
X = df[features]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("✅ Mean Squared Error:", mse)

# Save model
model_dir = './model'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'Regmodel.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Model saved successfully at {model_path}")
