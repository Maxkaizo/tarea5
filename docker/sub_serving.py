import pickle
from flask import Flask
from flask import Flask, request, jsonify

with open('model2.bin', 'rb') as model_file, open('dv.bin', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    dv = pickle.load(vectorizer_file)

def predict_single(customer, dv, model):
  X = dv.transform([customer])  ## apply the one-hot encoding feature to the customer data 
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred[0]

app = Flask('predict')
@app.route('/predict', methods=['POST'])  ## in order to send the customer information we need to post its data.
def predict():
    customer = request.get_json()  ## web services work best with json frame, So after the user post its data in json format we need to access the body of json.
    prediction = predict_single(customer, dv, model)
    subscription = prediction >= 0.5
    result = {
    'subscription probability': float(prediction), ## we need to cast numpy float type to python native float type
    'subscription': bool(subscription),  ## same as the line above, casting the value using bool method
    }
    return jsonify(result)  ## send back the data in json format to the user

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)