from fastapi import FastAPI,status,HTTPException
import pandas as pd
import numpy as np 
from pydantic import BaseModel
from utilities import scaling_data,upload,fit,predict,hotEncode
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from pathlib import Path

app = FastAPI()
class Data(BaseModel):
    Bhk:int
    HouseType : str
    sqft:float
    location : str

# Serve static files (CSS/JS if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path("templates/index.html")
    return html_path.read_text(encoding="utf-8")

@app.post("/predict")
def predictWeb(input:Data):
    location = input.location.strip().title() 
    try:
        coefficient = pd.read_csv("./Data/coefficient.csv")
        coefficient = coefficient[coefficient["City"]==location]
        w = coefficient["Weights"]
        b= coefficient["bias"]
    except:
        data = pd.read_csv("./Data/finalData.csv")
       
        data_city = data[data["location"]== location]
        
        X = data_city.drop(columns=["location","pricepersqft","totalprice"])
        
        y = data_city["totalprice"]
        
        X,waste,mean,std  = scaling_data(X,None)
       
        w,b,cost_history = fit(X,y,np.arange(X.shape[1]),0,30e-3, 1000)
        upload(input.location,w,b,mean,std)
   
    if input.HouseType.lower() == "villa":
        predictData = [input.Bhk,input.sqft,0,0,1]
    elif input.HouseType.lower() == "flat":
        predictData = [input.Bhk,input.sqft,1,0,0]
    else:
        predictData = [input.Bhk,input.sqft,0,1,0]

    for i in range(len(predictData)):
        predictData[i] = (predictData[i]-mean[i])/std[i]

    price = predict(predictData,w,b)
    return float(price)



