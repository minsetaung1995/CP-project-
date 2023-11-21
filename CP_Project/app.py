from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import pickle
import joblib
import numpy


app = Flask(__name__)

# Load the trained model (replace 'trained_model.pkl' with your model file)
with open('model/trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html', prediction_result='')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    input_features = request.form['input_features']
    
    # Perform data preprocessing on the input (if needed)
    input_data = pd.Series(input_features.split(','), index=['year', 'average_age', ...])
    
    # Make a prediction using the loaded model
    prediction = model.predict([input_data])

    # Convert the prediction to a human-readable result
    prediction_result = "Complete Game" if prediction == 1 else "Shutout"

    return render_template('index.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
