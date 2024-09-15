from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            overall_qual = int(request.form['overallQual'])
            exter_qual = int(request.form['exterQual'])
            bsmt_qual = int(request.form['bsmtQual'])
            total_bsmt_sf = float(request.form['totalBsmtSF'])
            first_flr_sf = float(request.form['1stFlrSF'])
            gr_liv_area = float(request.form['grLivArea'])
            garage_cars = int(request.form['garageCars'])
            garage_area = float(request.form['garageArea'])

            input_features = np.array([[overall_qual, exter_qual, bsmt_qual, total_bsmt_sf, first_flr_sf, gr_liv_area, garage_cars, garage_area]])
            input_features_scaled = scaler.transform(input_features)
            prediction = model.predict(input_features_scaled)
            predicted_price = round(prediction[0], 2)
            return render_template('index.html', prediction_text=f'Estimated House Price: ${predicted_price}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
