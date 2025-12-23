import os
import sys
from sklearn.metrics import r2_score
import dill
from src.exception import CustomException   
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)\
        

def evaluate_model(X_train, y_train, X_test, y_test, models,params):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]

            gs=GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Predicting the training set
            y_train_pred = model.predict(X_train)

            # Predicting the test set
            y_test_pred = model.predict(X_test)

            # Calculate r2 score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        logging.error("Error occurred during model evaluation")
        raise CustomException(e, sys)
    
def load_object(file_path):  
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.error("Error occurred while loading object")
        raise CustomException(e, sys) 