from fastapi import FastAPI
from modelInterface import ModelInterface

#fast api object declaration
app = FastAPI()
modelInterface=ModelInterface()


# expose methods through the class instance
@app.post("/predict")
def predict(txt:str):
    return str(modelInterface.predict(txt)[0])

@app.post("/predictHTML")
def predict_HTML(html):
    return str(modelInterface.predict(html)[0])
@app.get("/summary")
def model_summary()->str:
    modelInterface.model_summary()
    return "View server log for summary"
