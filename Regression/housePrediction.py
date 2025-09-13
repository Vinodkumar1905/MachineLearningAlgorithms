from fastapi import FastAPI,B


app = FastAPI()
class data(BaseModel):

@app.get("/predict")
def predict(data)