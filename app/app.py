from flask import Flask, request, render_template
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the trained model and other resources
model = pickle.load(open('../model.pkl', 'rb'))
region_mapping = {'Region1': 1, 'Region2': 2, 'Region3': 3}  # Map RegionName to numeric values

@app.route('/')
def index():
    # Render the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.form)
        # Get input values from the form
        region_name = request.form['RegionName']
        date = request.form['date']

        # Map RegionName to its numeric value
        region_numeric = int(region_name)

        # Convert the date to numeric days since the minimum date
        date_numeric = (pd.to_datetime(date) - pd.to_datetime("2000-01-01")).days

        # Create input DataFrame with the required features
        X_input = pd.DataFrame([[region_numeric, date_numeric]], columns=['RegionName', 'Date'])

        # Predict the Home Value Index
        prediction = model.predict(X_input)

        # Return the result
        return f"Predicted Home Value Index: {prediction[0]}"
    except KeyError as e:
        return f"Error: Invalid input value - {str(e)}"
    except Exception as e:
        return f"An error occurred - {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
