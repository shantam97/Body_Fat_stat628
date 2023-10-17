import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import joblib  # Added for saving sklearn models and transformers


sns.set()

data=pd.read_csv('BodyFat.csv')
data.columns = map(str.lower, data.columns)

# Dropping IDNO from the data 
data = data.drop(columns=['idno','density'])

# converting height from inches to m 
#data['height_m'] = data.height * 0.025

# converting weight from lbs to kg
#data['weight_kg'] = data.weight / 2.2
#data.weight_kg = round(data.weight_kg, 2)

# Converting all circumferential measures to meters, and rounding them off to 2 precision points
for col in data.columns[6:-1]:
    data[col] /= 100
    data[col] = round(data[col], 2)

#data.drop(columns=['weight','height'],axis=1,inplace=True)

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

final_data = data[~((data < (Q1 - 2 * IQR)) | (data > (Q3 + 2 * IQR))).any(axis=1)]

predictors = ['abdomen', 'adiposity', 'hip']

# Defining the dependent and independent variables 
y = final_data['bodyfat']
# x = final_data.drop(['bodyfat','age','ankle', 'bmi'],axis=1)
x = final_data[predictors]

# Yeo-Johnson Power Transformation inflates low variance data and deflates high variance data to create a more uniform dataset.
# This transformation helps in normalizing weightage of representation.
# This avoids any additional work needed for choosing test data. It can be chosen at random.
trans = PowerTransformer()
x = trans.fit_transform(x)
X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=42)


joblib.dump(trans, 'transformer.pkl')

# Modeling using Ridge Linear Regressor
ridge_params = {'alpha': [0.001,0.01, 0.1, 1.0, 5.0,10.0, 11.0, 12.0,13.0, 14.0, 15.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],  # Algorithm to use in the computation
                'max_iter': [100, 300, 500] 
               ,'random_state':[42]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=10, scoring='r2', n_jobs=-1)
ridge_grid.fit(X_train, y_train)
best_ridge_params = ridge_grid.best_params_

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
R2_score=r2_score(y_test,y_pred) * 100
RMSE=np.sqrt(mean_squared_error(y_test,y_pred))

print(f"Modeling Score using Ridge Linear Regressor is: \n{R2_score = }%, and \n{RMSE = }")

joblib.dump(model, 'model.pkl')