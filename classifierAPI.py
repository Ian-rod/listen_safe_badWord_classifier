from fastapi import FastAPI
from modelInterface import ModelInterface

#fast api object declaration
app = FastAPI()
modelInterface=ModelInterface()


# expose methods through the class instance
@app.post("/predict")
def predict(txt)->int:
    return modelInterface.predict(txt)

@app.post("/predictHTML")
def predict_HTML(html)->int:
    return modelInterface.predict(html)

@app.get("/summary")
def model_summary()->str:
    modelInterface.model_summary()
    return "View server log for summary"
