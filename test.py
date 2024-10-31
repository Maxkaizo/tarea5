import pickle

with open('model1.bin', 'rb') as model_file, open('dv.bin', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    dv = pickle.load(vectorizer_file)

def predict_single(customer, dv, model):
  X = dv.transform([customer])  ## apply the one-hot encoding feature to the customer data 
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred[0]

customer = {"job": "management", "duration": 400, "poutcome": "success"}

sub_probability = predict_single(customer, dv, model)
print('input', customer)
print('sub_probability', sub_probability)