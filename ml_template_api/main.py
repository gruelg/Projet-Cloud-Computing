import ml.model

import joblib
from fastapi import FastAPI
from datetime import date
date = date.today().isoformat()
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/{stri}")
def param( stri: str):

    model_filename = "model_{}.joblib.z".format(date)
    clf = joblib.load(model_filename)
    # Receives the input query from form
    probas = clf.coef_
    return {probas[0]}
