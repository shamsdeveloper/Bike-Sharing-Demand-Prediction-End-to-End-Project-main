from flask import Flask, render_template, request
import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return render_template('predicted_demand.html', demand=None)
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define required fields and their types
        required_fields = {
            'season': int, 'yr': int, 'mnth': int, 'hr': int, 'holiday': int,
            'weekday': int, 'workingday': int, 'weathersit': int,
            'temp': float, 'atemp': float, 'hum': float, 'windspeed': float
        }
        # Collect and validate input data
        input_data = {}
        for field, field_type in required_fields.items():
            value = request.form.get(field)
            if value is None or value.strip() == "":
                raise ValueError(f"Missing value for {field}")
            try:
                input_data[field] = field_type(value)  # Convert to the required type
            except ValueError:
                raise ValueError(f"Invalid value for {field}. Expected {field_type.__name__}.")

        input_df = pd.DataFrame([input_data])

        # Load training data for scaler reference
        train_df = pd.read_csv(os.path.join('artifacts', 'train.csv'))
        target_column_name = 'cnt'
        drop_columns = [target_column_name, 'instant', 'dteday', 'casual', 'registered']

        input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)

        # Scale the input data
        scaler = StandardScaler()
        scaler.fit(input_feature_train_df)  # Fit on training features
        input_feature_test_arr = scaler.transform(input_df)

        # Load the trained model
        model_path = os.path.join("artifacts", "model.pkl")
        loaded_model = pickle.load(open(model_path, 'rb'))

        # Perform prediction
        y_preds = loaded_model.predict(input_feature_test_arr)

        # Return result
        return render_template('predicted_demand.html', demand=round(y_preds[0], 0))

    except ValueError as ve:
        # Log the error and show user-friendly error messages
        print(f"Validation error: {ve}", file=sys.stderr)
        return render_template('predicted_demand.html', demand="Invalid input. Please check your values.")
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}", file=sys.stderr)
        return render_template('predicted_demand.html', demand="An error occurred during prediction.")


@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
