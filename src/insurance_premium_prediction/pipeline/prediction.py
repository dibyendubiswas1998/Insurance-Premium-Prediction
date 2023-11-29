import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model/model.joblib'))

    
    def prediction(self, data):
        try:
            prediction = self.model.predict(data)
            return prediction
        
        except Exception as ex:
            raise ex



if __name__ == "__main__":
    prd = PredictionPipeline()
    data = [[19, 0, 27.9, 0, 0]]
    result = prd.prediction(data=data)
    print(result)
    