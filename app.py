from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Dict

app = FastAPI()


model = joblib.load('best_model.pkl')
poly = joblib.load('poly.pkl')
scaler = joblib.load('scaler.pkl')


class PredictRequest(BaseModel):
    Annual_Rainfall: float
    Crop_Barley: float
    Crop_Banana: float
    Area: float
    Crop_Cashewnut: float
    Fertilizer : float
    
@app.post('/predict')
async def predict(request: PredictRequest):
    data = request.dict()
    

    df_input = pd.DataFrame([data])
    
   
    required_features = ['Annual_Rainfall', 'Area', 'Fertilizer'] 
    missing_features = [feature for feature in required_features if feature not in df_input.columns]
    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing_features)}")

  
    numeric_columns = df_input.select_dtypes(include=np.number).columns
    df_input[numeric_columns] = scaler.transform(df_input[numeric_columns])
    
    
    if 'Fertilizer' in df_input.columns and 'Annual_Rainfall' in df_input.columns:
        poly_features = poly.transform(df_input[['Annual_Rainfall', 'Fertilizer']])
        poly_features_df = pd.DataFrame(poly_features, 
                                        columns=poly.get_feature_names_out(['Annual_Rainfall', 'Fertilizer']))
        df_input = pd.concat([df_input, poly_features_df], axis=1)

    
    df_input_encoded = pd.get_dummies(df_input, drop_first=True)

   
    model_features = model.feature_names_in_ 
    missing_cols = set(model_features) - set(df_input_encoded.columns)
    for col in missing_cols:
        df_input_encoded[col] = 0
    df_input_encoded = df_input_encoded[model_features]

    
    prediction = model.predict(df_input_encoded)

    return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8001, log_level='debug')

