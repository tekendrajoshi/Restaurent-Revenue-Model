from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load(os.path.join(os.path.dirname(__file__), 'revenue_model.pkl'))
location_encoder = joblib.load('location_encoder.pkl')
cuisine_encoder = joblib.load('cuisine_encoder.pkl')
parking_encoder = joblib.load('parking_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get and validate form data
            form_data = {
                'Location': request.form['Location'],
                'Cuisine': request.form['Cuisine'],
                'Rating': float(request.form['Rating']),
                'Seating_Capacity': int(request.form['Seating_Capacity']),
                'Average_Meal_Price': float(request.form['Average_Meal_Price']),
                'Marketing_Budget': float(request.form['Marketing_Budget']),
                'Social_Media_Followers': int(request.form['Social_Media_Followers']),
                'Chef_Experience_Years': int(request.form['Chef_Experience_Years']),
                'Number_of_Reviews': int(request.form['Number_of_Reviews']),
                'Avg_Review_Length': float(request.form['Avg_Review_Length']),
                'Ambience_Score': float(request.form['Ambience_Score']),
                'Service_Quality_Score': float(request.form['Service_Quality_Score']),
                'Parking_Availability': request.form['Parking_Availability'],  # As string
                'Weekend_Reservations': int(request.form['Weekend_Reservations']),
                'Weekday_Reservations': int(request.form['Weekday_Reservations'])
            }

            # Encode categorical features
            encoded_data = {
                'Rating': form_data['Rating'],
                'Seating Capacity': form_data['Seating_Capacity'],
                'Average Meal Price': form_data['Average_Meal_Price'],
                'Marketing Budget': form_data['Marketing_Budget'],
                'Social Media Followers': form_data['Social_Media_Followers'],
                'Chef Experience Years': form_data['Chef_Experience_Years'],
                'Number of Reviews': form_data['Number_of_Reviews'],
                'Avg Review Length': form_data['Avg_Review_Length'],
                'Ambience Score': form_data['Ambience_Score'],
                'Service Quality Score': form_data['Service_Quality_Score'],
                'Weekend Reservations': form_data['Weekend_Reservations'],
                'Weekday Reservations': form_data['Weekday_Reservations'],
                'location_encoded': location_encoder.transform([form_data['Location']])[0],
                'cuisine_encoded': cuisine_encoder.transform([form_data['Cuisine']])[0],
                'parking_encoded': parking_encoder.transform([form_data['Parking_Availability']])[0]
            }

            # Convert to DataFrame with EXACT training columns
            input_df = pd.DataFrame([encoded_data], columns=[
                'Rating', 'Seating Capacity', 'Average Meal Price', 'Marketing Budget',
                'Social Media Followers', 'Chef Experience Years', 'Number of Reviews',
                'Avg Review Length', 'Ambience Score', 'Service Quality Score',
                'Weekend Reservations', 'Weekday Reservations', 'location_encoded',
                'cuisine_encoded', 'parking_encoded'
            ])

            # Predict
            prediction = model.predict(input_df)[0]
            return render_template("index.html", prediction=round(prediction, 2))

        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)