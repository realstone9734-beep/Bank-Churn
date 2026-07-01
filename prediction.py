def predict_churn(model, data):
    prediction = model.predict_proba(data)[0][1]
    return prediction