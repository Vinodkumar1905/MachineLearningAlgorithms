from fastapi import FastAPI,status,HTTPException
import pandas as pd
import numpy as np 
from pydantic import BaseModel
from utilities import scaling_data,upload,fit,predict,testTrainSplit
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


# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path("templates/index.html")
    return html_path.read_text(encoding="utf-8")

@app.post("/predict")
def predictWeb(input:Data):
    location = input.location.strip().title() 
    try:
        data = pd.read_csv("./Data/coefficient.csv")
        coefficient = data[data["City"]==location]
       
        row = coefficient.iloc[0]
     
        w_str = row["Weights"].strip('[]')
        w = np.array([float(x.strip()) for x in w_str.split(',')])
       
        mean_str = row["mean"].strip('[]')
        mean = np.array([float(x.strip()) for x in mean_str.split(',')])
        
        std_str = row["std"].strip('[]')
        std = np.array([float(x.strip()) for x in std_str.split(',')])
        
        b = float(row["bias"])
        print(f"Loaded coefficients for {location}: w shape {w.shape}, mean shape {mean.shape}")

    except:
        data = pd.read_csv("./Data/finalData.csv")
        data_city = data[data["location"]== location]
        
        X = data_city.drop(columns=["location","pricepersqft","totalprice"])
        
        y = data_city["totalprice"]
        X_tarin,waste1,y,waste = testTrainSplit(X,y,0.8,10)
        X,waste,mean,std  = scaling_data(X_tarin,None)
        w,b,cost_history = fit(X,y,np.zeros(X.shape[1]),0,30e-3, 1000)
        upload(input.location,w,b,mean,std)
   
    if input.HouseType.lower() == "villa":
        predictData = [input.Bhk,input.sqft,0,0,1]
    elif input.HouseType.lower() == "flat":
        predictData = [input.Bhk,input.sqft,1,0,0]
    else:
        predictData = [input.Bhk,input.sqft,0,1,0]
    print(predictData)
    print(mean)
    print(std)
    for i in range(len(predictData)):
        if std[i] == 0:
            std[i] = 1 
        predictData[i] = (predictData[i]-mean[i])/std[i]

    price = predict(predictData,w,b)
    return float(price)



