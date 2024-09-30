import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
import lime
import lime.lime_tabular
import shap
import numpy as np

app = Flask(__name__)

dt_model = pickle.load(open('DT_model.pkl', 'rb'))
adab_model = pickle.load(open('Adaboost-model.pkl', 'rb'))
# rf_model = pickle.load(open('RandomForest-model', 'rb'))

# Route for the Home Page (UI)
@app.route('/')
def home():
    return render_template('index.html')

# Route for Predicting and Showing LIME/SHAP Results
@app.route('/predict', methods=['POST'])
def predict():
    model_selected = request.form['model']
    if model_selected == 'model1':
        model = dt_model
        feature_names = ['review_count_x'
                        ,'avg_review_stars'              
                        ,'review_count_per_bussiness'    
                        ,'checkin_count'                 
                        ,'tip_count'                     
                        ,'total_compliments'             
                        ,'RestaurantsDelivery'           
                        ,'OutdoorSeating'                
                        ,'BusinessAcceptsCreditCards'    
                        ,'BikeParking'                   
                        ,'WiFi'                          
                        ,'Caters'                        
                        ,'WheelchairAccessible'          
                        ,'BusinessParking'               
                        ,'RestaurantsPriceRange2'        
                        ,'weekly_hours']
        
    # elif model_selected == 'model2':
    #     model = rf_model
    
    elif model_selected == 'model3':
        model = adab_model
        feature_names = ['review_count_x'
                        ,'avg_review_stars'              
                        ,'review_count_per_bussiness'
                        ,'avg_fans_per_reviewer'    
                        ,'checkin_count'                 
                        ,'tip_count'                     
                        ,'total_compliments'             
                        ,'RestaurantsDelivery'           
                        ,'OutdoorSeating'                
                        ,'BusinessAcceptsCreditCards'    
                        ,'BikeParking'
                        ,'RestaurantsTakeOut'                   
                        ,'WiFi'                          
                        ,'Caters'                        
                        ,'WheelchairAccessible'          
                        ,'BusinessParking'               
                        ,'RestaurantsPriceRange2'        
                        ,'weekly_hours']
    # Get data from the form
    data = [float(request.form[feature]) if request.form[feature] != '' else 0.0 for feature in feature_names]
    
    # Convert input into DataFrame
    input_data = pd.DataFrame([data], columns=feature_names)

    # Make predictions
    prediction = model.predict(input_data)[0]

    # Return result with LIME and SHAP visualizations
    return render_template('result.html', prediction=int(prediction))

if __name__ == '__main__':
    app.run(debug=True)