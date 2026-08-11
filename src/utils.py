import os
import sys
from src.exception import CustomException
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for name,model in models.items():
            model.fit(X_train,y_train)
            y_test_pred = model.predict(X_test)
            score = r2_score(y_test,y_test_pred)
            report[name] = score
        return report
    except Exception as e:
        raise CustomException(e,sys)