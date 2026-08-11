import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Splitting of data is completed")
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            logging.info("Model evaluation is initiated")
            report = evaluate_models(X_train, y_train, X_test, y_test, models)
            best_model_name = max(report, key=report.get)
            best_model_score = report[best_model_name]
            logging.info("Model evaluation is completed")
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            best_model = models[best_model_name]
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            prdicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,prdicted)
            logging.info(f"Model training is completed, R2 Score: {r2_square}")
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)