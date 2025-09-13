from fastapi import FastAPI,status,HTTPException
import pandas as pd
import numpy as np 
from utilities import scaling_data,upload,fit,predict

app = FastAPI()
class Data(BaseModel):
    Bhk:int
    HouseType : str
    sqft:float
    location : str

@app.get("/predict",status_code = status.HTTP_204)
def predict(input:Data):
    try:
        coefficient = pd.read("./Data/coefficient.csv")
        coefficient = coefficient[coefficient["City"]==input.location]
        w = coefficient["Weights"]
        b= coefficient["bias"]
    except:
        data = pd.read_csv("./Data/finalData.csv")
        X = data.drop(columns="totalprice")
        X = X[X["location"]=="Ahmedabad"]
        X = X.drop(columns=["location","pricepersqft"])
        y = data["totalprice"]
        X,waste,mean,std  = scaling_data(X,None)
        w,b,cost_history = fit(X,y,np.arrange(data.shape[1]),0,30e-3, 1000)
        upload(input.location,w,b,mean,std)
    inputData = pd.DataFrame(input)
    if input.HouseType.lower() == "villa":
        predictData = [input.Bhk,input.sqft,0,0,1]
        predictData = predictData
    elif input.HouseType.lower() == "flat":
        predictData = [input.Bhk,input.sqft,1,0,0]
    else:
        predictData = [input.Bhk,input.sqft,0,1,0]
    
    return predict(predictData,w,b)



