from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import pickle
import joblib
import numpy

app = Flask(__name__,template_folder='templates')


model = joblib.load('C:/Users/dell/Documents/learn Numpy/CP_Project/deployment/rfmodel1.pkl')
labels=float(['win_loss_percentage','ERA'])

@ app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

@ app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
      
    print(features)
    labels = model.predict([features])
    results = labels[0]
    if results<=4.0:
        r="It can be No1"
    else :
        r="It will not get no1"
    return r
        
  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

