from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo_02.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        CostCultivation = float(request.form['CostCultivation'])
        CostCultivation2 = float(request.form['CostCultivation2'])
        RainFall_Annual = float(request.form['RainFall_Annual'])
        Yield = float(request.form['Yield'])

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[CostCultivation, CostCultivation2, RainFall_Annual, Yield]], 
                               columns=['CostCultivation', 'CostCultivation2', 'RainFall Annual', 'Yield'])
        
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        prediction_value = float(prediction[0])  # Convertir el valor a tipo flotante
        
        app.logger.debug(f'Predicción: {prediction_value}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction_value})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

