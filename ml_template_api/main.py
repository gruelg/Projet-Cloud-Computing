import joblib
from fastapi import FastAPI
from datetime import date
import numpy as np
date = date.today().isoformat()
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/{stri}")
def param( stri: int):

    model_filename = "model_{}.joblib.z".format(date)
    clf = joblib.load(model_filename)
    return {"coef NA_Sales " :  clf.coef_[0], "coef EU_Sales " :  clf.coef_[1],"coef JP_Sales " :  clf.coef_[2],"coef Other_Sales " :  clf.coef_[3] }
