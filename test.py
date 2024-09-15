import unittest
import numpy as np
import pandas as pd
import warnings
import app


class FlaskTestCase(unittest.TestCase):
    # Ensure Flask app is created
    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

    # Test index route is accessible without login
    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    # Test POST request with correct input
    def test_valid_post(self):
        response = self.app.post('/', data={
            'overallQual': '7',
            'exterQual': '4',
            'bsmtQual': '4',
            'totalBsmtSF': '1000',
            '1stFlrSF': '1200',
            'grLivArea': '1500',
            'garageCars': '2',
            'garageArea': '500'
        })
        self.assertIn(b'Estimated House Price:', response.data)
        self.assertEqual(response.status_code, 200)

    # Test POST request with incomplete input
    def test_incomplete_post(self):
        response = self.app.post('/', data={
            'overallQual': '7',
            'exterQual': '4'
            # Missing other fields
        })
        self.assertIn(b'Error:', response.data)
        self.assertEqual(response.status_code, 200)

    # Test the loading of model and scaler
    def test_model_and_scaler_loaded(self):
        self.assertIsNotNone(app.model)
        self.assertIsNotNone(app.scaler)

    # Test the prediction output type and value sanity
    def test_prediction_output(self):
        with app.app.app_context():
            # Suppress specific sklearn warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Define features with names
                features = ['OverallQual', 'ExterQual', 'BsmtQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageCars', 'GarageArea']
                input_features = pd.DataFrame([[7, 4, 4, 1000, 1200, 1500, 2, 500]], columns=features)
                
                # Now using DataFrame instead of NumPy array to maintain feature names
                input_features_scaled = app.scaler.transform(input_features)
                prediction = app.model.predict(input_features_scaled)
                self.assertIsInstance(prediction, np.ndarray)
                self.assertGreater(prediction[0], 0)  # Assuming house prices cannot be zero or negative


if __name__ == '__main__':
    unittest.main()
