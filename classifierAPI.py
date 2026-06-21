from fastapi import FastAPI
from modelInterface import ModelInterface
from fastapi import Body
#fast api object declaration
app = FastAPI()
modelInterface=ModelInterface()


# expose methods through the class instance
@app.post("/predict")
async def predict(text: str = Body(...)):
    return float(modelInterface.predict(text)[0])

@app.post("/predictHTML")
def predict_HTML(html:str = Body(...)):
    return float(modelInterface.predict_Html(html)[0])
@app.get("/summary")
def model_summary()->str:
    modelInterface.model_summary()
    return "View server log for summary"
