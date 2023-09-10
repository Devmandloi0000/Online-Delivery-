import pandas as pd
import numpy as np
import os 
import sys

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_model
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:

    def __init__(self):
        self.Model_Trainer_Config_Path = ModelTrainerConfig()

    def Initiate_Model_Trainer(self,train_array,test_array):
        logging.info("starting of model training")
        try:
            # here in this we split data DEPENDENT AND INDEPENDENT
            X_train=train_array[:,:-1]
            X_test = test_array[:,:-1]
            y_train = train_array[:,-1]
            y_test = test_array[:,-1]

            #logging.info(X_train)
            #logging.info(y_train)
            #logging.info(X_test)
            #logging.info(y_test)
            
            logging.info("Now model Training Processes has Started")
            models = {
                    'LinearRegression':LinearRegression(),
                    'Lasso':Lasso(),
                    'ElasticNet':ElasticNet(),
                    'Ridge':Ridge()
                }
            
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            #print(model_report)
            print("\n===============================================================================\n")
            logging.info(f"{model_report}")

            # sorting model based on there performance
            best_model_score = max(sorted(model_report.values()))
            logging.info(f"best model score is :- {best_model_score}")

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                self.Model_Trainer_Config_Path.trained_model_path,
                obj= best_model
            )
                

            
        except Exception as e:
            logging.info("there may be some error in initiate model trainer")
            raise CustomException(e,sys)