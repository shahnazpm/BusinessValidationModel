import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
import lime
import lime.lime_tabular
import shap
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

# Route for the Home Page (UI)
@app.route('/')
def home():
    return render_template('index.html')

# Route for Predicting and Showing LIME/SHAP Results
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = [float(request.form[feature]) for feature in feature_names]
    
    # Convert input into DataFrame
    input_data = pd.DataFrame([data], columns=feature_names)

    # Make predictions
    prediction = model.predict(input_data)[0]

    # Return result with LIME and SHAP visualizations
    return render_template('result.html', prediction=int(prediction))

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)

#     input_data = pd.DataFrame([data], columns=feature_names)

#     prediction = model.predict(input_data)[0]

#     # lime_exp = explainer.explain_instance(input_data.values[0], model.predict_proba, num_features=2)
#     # lime_html = lime_exp.as_html()

#     # shap_values = shap_explainer.shap_values(input_data)
#     # shap_summary = shap.force_plot(shap_explainer.expected_value[1], shap_values[1], input_data)

#     return jsonify({
#         'prediction': int(prediction),
#         'lime_interpretation': lime_html,
#         'shap_interpretation': str(shap_summary)
#     })

if __name__ == '__main__':
    app.run(debug=True)