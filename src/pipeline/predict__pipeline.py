# a simple web server to serve predictions
import os
import sys
import pandas as pd
from src.exception import CustomException       
from src.logger import logging
from src.utils import load_object
#preprocessor is responsible for handling categorical and numerical features

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            logging.info("Loading preprocessor and model")
            model =load_object(file_path=model_path)
            preprocessor =load_object(file_path=preprocessor_path)
            logging.info("Transforming features using preprocessor")
            data_scaled = preprocessor.transform(features)
            logging.info("Making predictions")
            preds = model.predict(data_scaled)
            return preds
        except Exception as e: 
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)
class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course  
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_data_frame(self): 
        try:
            custom_data_input_dict ={
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education], 
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.error("Error occurred while converting data to DataFrame")
            raise CustomException(e, sys)