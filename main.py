import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
scaler = pickle.load(open('scaler.pkl','rb'))
knn_model = pickle.load(open('knn_model.pkl','rb'))
nn_model = pickle.load(open('nn_model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    data=pd.DataFrame({"Rooms": [request.form.get('rooms')], "Distance": [request.form.get('distance')], "Bathroom": [request.form.get('bathroom')], "Landsize": [request.form.get('landsize')], "BuildingArea": [request.form.get('buildingarea')], "YearBuilt": [request.form.get('yearbuilt')]})
    scaled = scaler.transform(data)
    knn_prediction = knn_model.predict(scaled).reshape(-1, 1)
    nn_prediction = nn_model.predict(knn_prediction)
    result = round(nn_prediction[0][0],2)
    return render_template('index.html', prediction_text=f'Price of House is Rs. {result}/-')

if __name__ == '__main__':
    app.run(debug=True)