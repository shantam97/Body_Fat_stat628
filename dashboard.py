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



# Ensure your previous code is intact here for model training...

# Importing necessary libraries for Dash app
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import traceback

# Dummy averages for visualization - replace with actual values
average_values = {'abdomen': 0.90, 'adiposity': 25, 'hip': 1.0, 'bodyfat': 18}

# Initialize Dash app
# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,]) 
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True 


# Inline styles
styles = {
    'container': {
        'backgroundColor': '#f8f9fa'
        
    },
    
    'header': {
        'color': '#0d6efd',
        'textAlign': 'center',
        'margin': '50px'
    },
    'sub_header': {
        'color': '#0d6efd'
    },
    'dropdown': {
        'width': '100%',  # This makes the dropdown match the width of its container
        'marginBottom': '20px'  # Optional: for space between the dropdown and the scatter plot
    },
    'scatter': {
        'width': '100%',  # This makes the scatter plot match the width of its container
        'marginTop': '20px'  # Optional: for space between the dropdown and the scatter plot
    },
    'button': {
        'width': '100%',
        'color': '#fff',
        'backgroundColor': '#0d6efd',
        'borderColor': '#0d6efd'
    },
    'graph': {
        'boxShadow': '0 1px 2px 0 rgba(0, 0, 0, 0.1)',
        'marginTop': '20px'
    }
}


app.layout = dbc.Container([
    dbc.Row(html.H1("Body Fat Prediction", className="text-center my-4")),  # Centered heading with margin
    dbc.Row([
        dbc.Col([
            html.H2("Input Features"),
            html.Div("Enter the following details :"),
            
            # Reformatted the layout without dbc.FormGroup
            dbc.Row([
                dbc.Col(dbc.Label("Abdomen (in cm):"), width=6),
                dbc.Col(dbc.Input(id='input-abdomen', type='number', value=85, min=60), width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col(dbc.Label("Height (in cm ):"), width=6),
                dbc.Col(dbc.Input(id='input-height', type='number', value=165, min=120), width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col(dbc.Label("Weight (in KG ):"), width=6),
                dbc.Col(dbc.Input(id='input-weight', type='number', value=70, min=30.0), width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col(dbc.Label("Hip (in cm):"), width=6),
                dbc.Col(dbc.Input(id='input-hip', type='number', value=95, min=80), width=6),
            ], className="mb-3"),
            
            dbc.Button('Predict', id='predict-button', color="primary", className="mt-3", n_clicks=0,style=styles['button']),
            html.H2(id='output-prediction', className="my-4",style=styles['sub_header']),
        ], width=4),
        
        dbc.Col([
            dcc.Dropdown(
                id='scatter-dropdown',
                options=[
                    {'label': 'Abdomen', 'value': 'abdomen'},
                    {'label': 'Adiposity', 'value': 'adiposity'},
                    {'label': 'Hip', 'value': 'hip'}
                ],
                value='abdomen',
             style=styles['dropdown'] # Default value
            ),
            dcc.Graph(id='scatter-plot',style=styles['scatter']),
        ], width=6),
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='feature-importance-plot'), width=6),
        dbc.Col(dcc.Graph(id='comparison-visualization'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='correlation-heatmap'), width=6),
        dbc.Col(dcc.Graph(id='residual-plot'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='feature-boxplot'),style=styles['graph'], width=6),
        dbc.Col(dcc.Graph(id='actual-predicted-plot'), width=6),
    ]),
    


], style=styles['container'],fluid=True)

@app.callback(
    [
        Output("output-prediction", "children"),
        Output("scatter-plot", "figure"),
        Output("comparison-visualization", "figure"),
        Output("feature-importance-plot", "figure"),
        Output("correlation-heatmap", "figure"),
        Output("residual-plot", "figure"),
        Output("feature-boxplot", "figure"),
        Output("actual-predicted-plot", "figure"),
         
    ],
    [
        Input("scatter-dropdown", "value"),  # corrected from "selected-feature"
        Input("predict-button", "n_clicks")  # corrected from "submit-button"
    ],
    [
        State("input-abdomen", "value"),  # corrected from "abdomen-input"
        State("input-height", "value"),
        State("input-weight", "value"),  # corrected from "adiposity-input"
        State("input-hip", "value")  # corrected from "hip-input"
    ],
)





def update_output(selected_feature, n_clicks, abdomen, height,weight,hip):
    
    if n_clicks is None:
        return 'Enter values and click Predict to get the body fat prediction.', {}, {}, {}
    try:
        abdomen_m = abdomen / 100
        hip_m = hip / 100
        height=height/100
        if height <= 0:
            raise ValueError("Height cannot be less than or equal to zero.")
        if weight <= 0:
         raise ValueError("Weight cannot be less than or equal to zero.")
        adiposity = weight / (height ** 2)
        # Prepare user input for prediction
        input_data = np.array([abdomen_m, adiposity, hip_m]).reshape(1, -1)
        input_data_transformed = trans.transform(input_data)
        prediction = model.predict(input_data_transformed)[0]
        
        prediction_text = f'Predicted Body Fat: {prediction:.2f}%'

    

        # Create various plots
        # Move the scatter plot update logic to its own function
        # Scatter plot logic
        
        # Identify the index of the selected feature in the model's coefficients
        selected_feature_index = predictors.index(selected_feature)

        # Get the coefficient (slope) for the selected feature
        slope = model.coef_[selected_feature_index]

        # Calculate the trendline (y = mx + c)
        x_range = np.linspace(final_data[selected_feature].min(), final_data[selected_feature].max(), 100)  # Generates 100 points between min and max
        trendline_y = slope * x_range + model.intercept_  # Applies the linear equation

        # Generate the scatter plot
        scatter_fig = px.scatter(final_data, x=selected_feature, y='bodyfat', title=f'{selected_feature.capitalize()} vs Body Fat')
        #scatter_fig.add_scatter(x=x_range, y=trendline_y, mode='lines', name='Trendline', line=dict(color='Black'))

        # Adding user input to the scatter plot
        user_input = abdomen_m if selected_feature == 'abdomen' else (adiposity if selected_feature == 'adiposity' else hip_m)
        scatter_fig.add_scatter(x=[user_input], y=[prediction], mode='markers', marker={'size': 15, 'color': 'red'}, name='Your Input')

        

        # Feature importance

        feature_importance = pd.DataFrame({
            'Feature': ['Abdomen', 'Adiposity', 'Hip'],
            'Importance': model.coef_
        })

        fig_feature_importance = px.bar(
            feature_importance, 
            x='Feature', y='Importance', 
            title='Feature Importance',
            labels={'Feature': 'Feature', 'Importance': 'Coefficient Value'}
            )
        fig_feature_importance.update_layout(yaxis=dict(title='Coefficient Value'))
        

        

        # Comparison visualization
        comparison_fig = go.Figure(data=[
                go.Bar(name='User Input', x=['Abdomen', 'Adiposity', 'Hip', 'Predicted Body Fat'], y=[abdomen, adiposity, hip, prediction]),
                go.Bar(name='Average', x=['Abdomen', 'Adiposity', 'Hip', 'Average Body Fat'], y=[average_values['abdomen']*100, average_values['adiposity'], average_values['hip']*100, average_values['bodyfat']])
            ])
        comparison_fig.update_layout(barmode='group', title='Comparison with Average Values')


    
        # Correlation Heatmap
        # Correlation Heatmap with a Red-Green color scale
        df_corr = final_data.corr().round(1)  # Assuming 'final_data' is your DataFrame
        mask = np.zeros_like(df_corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')

        # Create the correlation heatmap with text annotated inside
        fig_corr = px.imshow(
            df_corr_viz, 
            text_auto=True, 
            color_continuous_scale="RdYlGn",  # Red to Green diverging color scale
            title='Feature Correlation Matrix'
        )

        
        # Residual Plot
        residuals = y_test - y_pred
        residual_plot = px.scatter(x=y_pred, y=residuals)
        residual_plot.add_hline(y=0, line_dash="dash", line_color="red")
        residual_plot.update_layout(title='Residuals vs. Predicted', xaxis_title='Predicted', yaxis_title='Residuals')

        
        # Box Plots for Features
        box = go.Figure()
        
        for feature in predictors:
                box.add_trace(go.Box(y=final_data[feature], name=feature, boxpoints='all', jitter=0.5, whiskerwidth=0.2))
        box.update_layout(title='Distribution of Features')

        # Actual vs Predicted Value Plot
        actual_vs_predicted = go.Figure()
        actual_vs_predicted.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='data'))
        actual_vs_predicted.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='ideal fit'))
        actual_vs_predicted.update_layout(title='Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')

        return (prediction_text, scatter_fig, comparison_fig, fig_feature_importance, fig_corr, residual_plot, box, actual_vs_predicted)

    except Exception as e:
        print(e)  # This helps you understand the traceback
    return (f"An error occurred: {str(e)}", {}, {}, {})


if __name__ == '__main__':
    app.run_server(debug=True)





