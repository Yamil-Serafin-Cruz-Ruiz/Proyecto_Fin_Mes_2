from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import logging
import joblib

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Carga del modelo entrenado
model = joblib.load('network_neuronal.pkl')
scaler = joblib.load('scaler_6.pkl')

app.logger.debug('Modelo cargado correctamente')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los valores enviados
        features = [
            float(request.form['AT']),
            float(request.form['AH']),
            float(request.form['AFDP']),
            float(request.form['GTEP']),
            float(request.form['TAT']),
            float(request.form['CO']),
        ]

        app.logger.debug(f'Valores recibidos para predicción: {features}')

        input_array = np.array([features])
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)
        app.logger.debug(f'Predicción calculada: {prediction[0]}')

        return jsonify({'prediccion': float(prediction[0])})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {e}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
